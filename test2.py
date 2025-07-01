import weaviate
import os
from beir.datasets.data_loader import GenericDataLoader
from beir.util import download_and_unzip
from weaviate.classes.config import Property, DataType
from tqdm.autonotebook import tqdm
import requests
import time
import logging  # Import logging module for better error reporting

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
DATASET_NAME = "scifact"
DATA_PATH = "datasets"
WEAVIATE_HOST = "127.0.0.1"
WEAVIATE_PORT = 8080
WEAVIATE_GRPC_PORT = 50051
COLLECTION_NAME = "document_v4"
E5_INFERENCE_API_URL = "http://localhost:8081/vectors"  # Endpoint from your screenshot
# IMPORTANT: Adjust this key based on the actual JSON response from your /vectors endpoint.
# Common keys are 'vector', 'embedding', 'embeddings', 'data', etc.
# If `curl -X POST -H "Content-Type: application/json" -d '{"text": "test"}' http://localhost:8081/vectors`
# returns {"embedding": [0.1, 0.2, ...]}, then set this to 'embedding'.
# If it returns {"vector": [0.1, 0.2, ...]}, then set this to 'vector'.
# If it returns [0.1, 0.2, ...], then set this to None (meaning response.json() is directly the list).
E5_VECTOR_KEY_IN_RESPONSE = 'vector'  # <<< YOU MIGHT NEED TO CHANGE THIS BASED ON ACTUAL API RESPONSE

# --- 1. Download SciFact dataset ---
logging.info(f"Downloading and unzipping {DATASET_NAME} dataset...")
try:
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET_NAME}.zip"
    download_and_unzip(url, DATA_PATH)
    logging.info("Dataset downloaded.")
except Exception as e:
    logging.error(f"Failed to download or unzip dataset: {e}")
    exit(1)

# --- 2. Load dataset ---
logging.info(f"Loading {DATASET_NAME} dataset...")
try:
    corpus, queries, qrels = GenericDataLoader(os.path.join(DATA_PATH, DATASET_NAME)).load(split="test")
    logging.info(f"Dataset loaded. Corpus size: {len(corpus)}")
except Exception as e:
    logging.error(f"Failed to load dataset: {e}")
    exit(1)


# --- 3. E5 Inference API Function ---
def get_embedding_for_text(text: str):
    """
    Calls your E5 inference service to get a single embedding.
    Assumes API endpoint POST /vectors expects {"text": "string"}
    and extracts the vector based on E5_VECTOR_KEY_IN_RESPONSE.
    """
    try:
        payload = {"text": text}
        # Add config if you need specific pooling/task_type, e.g.,
        # payload["config"] = {"pooling_strategy": "mean", "task_type": "retrieval"}

        response = requests.post(E5_INFERENCE_API_URL, json=payload, timeout=60)  # Increased timeout
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

        json_response = response.json()

        if E5_VECTOR_KEY_IN_RESPONSE:
            # Extract vector from specific key
            vector = json_response.get(E5_VECTOR_KEY_IN_RESPONSE)
            if vector is None:
                logging.error(f"E5 API response missing key '{E5_VECTOR_KEY_IN_RESPONSE}': {json_response}")
                return None
        else:
            # Assume response.json() is directly the vector list
            vector = json_response

        if not isinstance(vector, list) and not (hasattr(vector, 'dtype') and 'float' in str(
                vector.dtype)):  # Check if it's a list or numpy array of floats
            logging.error(
                f"E5 API response vector is not a list of numbers. Type: {type(vector)}, Value: {str(vector)[:200]}...")
            return None

        return vector

    except requests.exceptions.Timeout:
        logging.error(f"E5 inference API request timed out for text (first 100 chars): '{text[:100]}...'")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling E5 inference API for text (first 100 chars): '{text[:100]}...': {e}")
        return None
    except json.JSONDecodeError:
        logging.error(
            f"E5 inference API returned non-JSON response for text (first 100 chars): '{text[:100]}...'. Response: {response.text}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error in get_embedding_for_text for text (first 100 chars): '{text[:100]}...': {e}")
        return None


# --- 4. Connect to Weaviate ---
logging.info("Connecting to Weaviate...")
client = None
try:
    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
    )
    client.is_live()
    logging.info("Successfully connected to Weaviate!")
except Exception as e:
    logging.critical(
        f"Could not connect to Weaviate. Please ensure your Docker container is running and accessible: {e}")
    exit(1)

# --- 5. Create/Recreate Collection ---
logging.info(f"Checking for existing collection '{COLLECTION_NAME}'...")
try:
    if client.collections.exists(COLLECTION_NAME):
        logging.info(f"Collection '{COLLECTION_NAME}' found. Deleting and recreating to ensure correct schema.")
        client.collections.delete(COLLECTION_NAME)
        time.sleep(1)  # Give Weaviate a moment to process the deletion

    logging.info(f"Creating collection '{COLLECTION_NAME}' with updated schema (no Weaviate vectorizer)...")
    client.collections.create(
        name=COLLECTION_NAME,
        description="Scientific documents from SciFact dataset with original BEIR doc_id and pre-computed E5 vectors",
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="original_doc_id", data_type=DataType.TEXT)  # Crucial for BEIR evaluation
        ],
        # No vectorizer_config here, as we are providing vectors directly
    )
    logging.info(f"Collection '{COLLECTION_NAME}' created successfully!")
except Exception as e:
    logging.critical(f"Failed to create/recreate Weaviate collection: {e}")
    client.close()
    exit(1)

# --- 6. Batch Insert Documents with Pre-computed Vectors ---
documents_collection = client.collections.get(COLLECTION_NAME)

logging.info("Starting batch ingestion with external embedding generation...")

num_inserted = 0
num_skipped = 0

with documents_collection.batch.dynamic() as batch:
    for doc_id, doc_data in tqdm(corpus.items(), desc="Ingesting documents"):
        text_to_embed = "passage: " + doc_data.get("text", "")

        # Get embedding for the current text
        current_vector = get_embedding_for_text(text_to_embed)

        if current_vector is not None:
            try:
                batch.add_object(
                    properties={
                        "original_doc_id": doc_id,
                        "title": doc_data.get("title", ""),
                        "text": doc_data.get("text", "")
                    },
                    vector=current_vector  # Provide the vector directly
                )
                num_inserted += 1
            except Exception as e:
                logging.error(f"Failed to add object {doc_id} to Weaviate batch: {e}")
                num_skipped += 1
        else:
            num_skipped += 1
            # Error message already logged by get_embedding_for_text

logging.info(f"Finished ingestion. Inserted {num_inserted} documents. Skipped {num_skipped} documents due to errors.")
if client:
    client.close()
    logging.info("Ingestion client closed.")
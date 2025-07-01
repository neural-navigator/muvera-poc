import weaviate
import os
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval  # Correct import for BEIR 2.2.0
from tqdm.autonotebook import tqdm
import requests
import json
import logging  # Import logging module for better error reporting

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
DATASET_NAME = "scifact"
DATA_PATH = "datasets"
WEAVIATE_HOST = "127.0.0.1"
WEAVIATE_PORT = 8080
WEAVIATE_GRPC_PORT = 50051
COLLECTION_NAME = "document_v4"  # Must match the ingestion script's collection name
E5_INFERENCE_API_URL_FOR_QUERY = "http://localhost:8081/vectors"  # Endpoint for query embedding
# IMPORTANT: This must match the key you determined for your E5 API response in the ingestion script.
# If `curl -X POST -H "Content-Type: application/json" -d '{"text": "test"}' http://localhost:8081/vectors`
# returns {"embedding": [0.1, 0.2, ...]}, then set this to 'embedding'.
# If it returns {"vector": [0.1, 0.2, ...]}, then set this to 'vector'.
# If it returns [0.1, 0.2, ...], then set this to None.
E5_VECTOR_KEY_IN_RESPONSE = 'vector'  # <<< ENSURE THIS MATCHES YOUR INGESTION SCRIPT'S VALUE

# --- 1. Load the SciFact dataset (for queries and qrels) ---
logging.info(f"Loading {DATASET_NAME} dataset for evaluation...")
try:
    corpus, queries, qrels = GenericDataLoader(os.path.join(DATA_PATH, DATASET_NAME)).load(split="test")
    logging.info(f"Dataset for evaluation loaded. Number of queries: {len(queries)}")
except Exception as e:
    logging.critical(f"Failed to load dataset for evaluation: {e}")
    exit(1)


# --- 2. E5 Inference API Function for Queries ---
# This function is identical to the one in the ingestion script, ensuring consistency.
def get_embedding_for_text(text: str):
    """
    Calls your E5 inference service to get a single embedding for a query.
    Extracts the vector based on E5_VECTOR_KEY_IN_RESPONSE.
    """
    try:
        payload = {"text": text}

        response = requests.post(E5_INFERENCE_API_URL_FOR_QUERY, json=payload, timeout=60)
        response.raise_for_status()

        json_response = response.json()

        if E5_VECTOR_KEY_IN_RESPONSE:
            vector = json_response.get(E5_VECTOR_KEY_IN_RESPONSE)
            if vector is None:
                logging.error(
                    f"E5 API response missing key '{E5_VECTOR_KEY_IN_RESPONSE}' for query (first 100 chars): '{text[:100]}...': {json_response}")
                return None
        else:
            vector = json_response

        if not isinstance(vector, list) and not (hasattr(vector, 'dtype') and 'float' in str(vector.dtype)):
            logging.error(
                f"E5 API response vector for query is not a list of numbers. Type: {type(vector)}, Value: {str(vector)[:200]}...")
            return None

        return vector

    except requests.exceptions.Timeout:
        logging.error(f"E5 inference API request timed out for query (first 100 chars): '{text[:100]}...'")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling E5 inference API for query (first 100 chars): '{text[:100]}...': {e}")
        return None
    except json.JSONDecodeError:
        logging.error(
            f"E5 inference API returned non-JSON response for query (first 100 chars): '{text[:100]}...'. Response: {response.text}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error in get_embedding_for_text for query (first 100 chars): '{text[:100]}...': {e}")
        return None


# --- 3. Connect to Weaviate ---
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

# Get the collection
try:
    documents_collection = client.collections.get(COLLECTION_NAME)
    logging.info(f"Successfully retrieved collection '{COLLECTION_NAME}'.")
except Exception as e:
    logging.critical(
        f"Failed to get Weaviate collection '{COLLECTION_NAME}'. It might not be created or populated correctly. Error: {e}")
    if client:
        client.close()
    exit(1)

# --- 4. Perform Searches in Weaviate for each query ---
logging.info(f"Performing searches in Weaviate collection '{COLLECTION_NAME}'...")
results = {}  # This will store results in the format expected by BEIR evaluator

k_values = [1, 3, 5, 10, 100]  # Define k values for evaluation metrics
max_k = max(k_values)

num_queries_processed = 0
num_queries_skipped = 0

for query_id, query_text in tqdm(queries.items(), desc="Searching Weaviate"):
    # First, get the embedding for the query
    query_vector = get_embedding_for_text("query: " + query_text)

    if query_vector is None:
        logging.warning(f"Skipping query {query_id} due to embedding error.")
        num_queries_skipped += 1
        continue  # Skip this query if embedding fails

    try:
        # Perform a near_vector search using the pre-computed query embedding
        response = documents_collection.query.near_vector(
            near_vector=query_vector,  # Pass the vector directly for search
            limit=max_k,  # Retrieve enough results to cover all k_values
            return_properties=["original_doc_id"],  # Crucial for mapping back to qrels
            return_metadata=weaviate.classes.query.MetadataQuery(score=True)  # Get scores for ranking
        )

        results[query_id] = {}
        for obj in response.objects:
            doc_id_from_weaviate = obj.properties.get("original_doc_id")
            if doc_id_from_weaviate:
                results[query_id][doc_id_from_weaviate] = obj.metadata.score
            else:
                logging.warning(
                    f"original_doc_id property not found for object {obj.uuid} from query {query_id}. Skipping this document for evaluation.")
        num_queries_processed += 1

    except Exception as e:
        logging.error(f"Error during Weaviate search for query {query_id}: {e}")
        num_queries_skipped += 1

logging.info(f"Search complete. Processed {num_queries_processed} queries, skipped {num_queries_skipped} queries.")
logging.info("Evaluating results...")

# --- 5. Evaluate the results using BEIR's `EvaluateRetrieval` class ---
if not results:
    logging.critical("No search results to evaluate. Please check ingestion and search steps.")
    if client:
        client.close()
    exit(1)

try:
    retriever = EvaluateRetrieval(k_values=k_values)
    ndcg, _map, _recall, _precision = retriever.evaluate(qrels, results, k_values)

    logging.info("\n--- Evaluation Results ---")
    logging.info(f"NDCG@{k_values}: {ndcg}")
    logging.info(f"MAP@{k_values}: {_map}")
    logging.info(f"Recall@{k_values}: {_recall}")
    logging.info(f"Precision@{k_values}: {_precision}")

except Exception as e:
    logging.critical(f"Error during BEIR evaluation: {e}")
finally:
    if client:
        client.close()
        logging.info("Weaviate client closed.")
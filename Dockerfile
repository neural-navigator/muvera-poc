FROM semitechnologies/transformers-inference:custom

# Required for sentence-transformers support (pooling, batching, etc.)
ENV USE_SENTENCE_TRANSFORMERS_VECTORIZER=true

# Download the E5 model
RUN MODEL_NAME=intfloat/e5-base-v2 ./download.py

# Green Language Model Inference: Impact of Retrieval-Augmented Generation

Setup:
- Install weaviate locally: `docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.26.5`
- Install qdrant locally: `docker pull qdrant/qdrant` + `docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant`
- Install milvus locally: https://milvus.io/docs/install_standalone-docker-compose.md - path: C:\Users\YourUser

Create embeddings:
- Run: `python ./embeddings/create_embeddings.py Alibaba-NLP/gte-base-en-v1.5 quora 32`, where the arguments are the model name, dataset and batch size
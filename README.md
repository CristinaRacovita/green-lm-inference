# Green Language Model Inference: Impact of Retrieval-Augmented Generation

Setup:
- Install weaviate locally: `docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.26.5`
- Install milvus locally:
 `docker pull milvusdb/milvus:v2.1.0` + `docker run -d --name milvus -p 19530:19530 -p 19121:19121 -v milvus_data:/var/lib/milvus milvusdb/milvus:v2.1.0`

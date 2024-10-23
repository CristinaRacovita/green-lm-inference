# Green Language Model Inference: Impact of Retrieval-Augmented Generation

This repository contains the code needed to replicate the experiments described in the research mentioned above paper. **The methodology was tested only on a Windows machine**.

### Setup
- Install weaviate locally: 
    - `docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.26.5`
- Install qdrant locally:
    - `docker pull qdrant/qdrant` + `docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant`
- Install milvus locally: 
    - https://milvus.io/docs/install_standalone-docker-compose.md - path: C:\Users\YourUser
- Create the Conda environment that contains the required Python packages:
    - `conda env create -f environment.yml`

### Research Questions
To monitor the used resources, before running any experiment, the HWiNFO logging has to be started. The measurements have to be stored in the results directory under the subdirectory that corresponds to the research question of interest. Please use the same filenames as the ones from this repository.

Before conducting the experiments, run `create_all_embeddings.sh`. This will create corpus and query embeddings for each dataset with each embedding model.

#### RQ1
Index or query 10 times each dataset of embeddings using Milvus DB: `rq1.sh index` and `rq1.sh query`.

#### RQ2
Index or query 10 times the cqadupstack-webmasters dataset vectorized with gte-base using each of the three data bases: `rq2.sh index` and `rq2.sh query`.

#### RQ3
Benchmark the embedding generation which creates embeddings for nfcorpus dataset 10 times with each embedding model: `rq3.sh`.

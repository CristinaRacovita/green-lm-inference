import sys
sys.path.append("../")

from utils import load_query_text
from vector_databases.db_operations import get_client, get_prompt_docs

DIRECTORY_PATH = "../data/datasets"
VECTOR_DB = "qdrant"
DATASET_NAME = ""

texts = load_query_text(DIRECTORY_PATH + "/arguana/queries.jsonl", 10)
client = get_client(VECTOR_DB)

embeddings = 

similar_retrieved_texts = get_prompt_docs(client, DATASET_NAME, 10, [0])

# load embedding model, language model, vector db

# get queries

# for each query: embed it, get from the vector db top k answers, create the prompt, generate answer

# before benchmarking answer to one question
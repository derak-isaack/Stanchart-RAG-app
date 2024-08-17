from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from haystack import Pipeline
from haystack_integrations.components.retrievers.opensearch import OpenSearchEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

app = FastAPI()

# Define the data model for the request body
class Query(BaseModel):
    query: str

# Initialize document store and query pipeline
document_store = OpenSearchDocumentStore(
    hosts=["http://localhost:9200"],
    use_ssl=False,
    verify_certs=False,
    http_auth=("admin", "admin")
)

model = "sentence-transformers/all-mpnet-base-v2"

query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model=model))
query_pipeline.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

@app.post("/search")
async def search(query: Query):
    result = query_pipeline.run({"text_embedder": {"text": query.query}})
    
    if result['retriever']['documents']:
        doc = result['retriever']['documents'][0]
        return {
            "content": doc.content,
            "metadata": doc.meta
        }
    else:
        return {"message": "No results found."}

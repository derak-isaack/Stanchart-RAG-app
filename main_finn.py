from haystack_integrations.components.retrievers.opensearch  import OpenSearchEmbeddingRetriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack import Document
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from langchain_community.document_loaders import PyMuPDFLoader

document_store = OpenSearchDocumentStore(hosts="http://localhost:9200", use_ssl=False,
verify_certs=False, http_auth=("admin", "admin"))

model = "sentence-transformers/all-mpnet-base-v2"


loader = PyMuPDFLoader("D:/projects/devposts/standard-chartered-plc-full-year-2023-report.pdf")
docs = loader.load()
documents = [docs]


document_embedder = SentenceTransformersDocumentEmbedder(model=model)  
document_embedder.warm_up()

haystack_documents = []
for doc in documents:
    for page in doc:
        haystack_documents.append(Document(content=page.page_content, meta=page.metadata))
    
documents_with_embeddings = document_embedder.run(haystack_documents)

# documents_with_embeddings = document_embedder.run(documents)

document_store.write_documents(documents_with_embeddings.get("documents"), policy=DuplicatePolicy.SKIP)

query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model=model))
query_pipeline.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

query = "what are the KPI's for stanchart?"

result = query_pipeline.run({"text_embedder": {"text": query}})

print(result['retriever']['documents'][0])
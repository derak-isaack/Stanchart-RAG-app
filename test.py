from haystack_integrations.components.retrievers.opensearch  import OpenSearchEmbeddingRetriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack import Document
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from langchain_community.document_loaders import PyMuPDFLoader

document_store = OpenSearchDocumentStore(
    hosts="http://localhost:9200", 
    use_ssl=False,
    verify_certs=False,
    http_auth=("Admin", "admin")
)

loader = PyMuPDFLoader("D:/projects/devposts/standard-chartered-plc-full-year-2023-report.pdf")
docs = loader.load()
documents = [docs]

haystack_documents = []
for doc in documents:
    for page in doc:
        haystack_documents.append(Document(content=page.page_content, meta=page.metadata))
    


if not document_store._search_documents:
    document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-mpnet-base-v2")
    document_embedder.warm_up()
    documents_with_embeddings = document_embedder.run(haystack_documents)
    
    document_store.write_documents(documents_with_embeddings.get("documents"), policy=DuplicatePolicy.SKIP)
    print("Documents stored in OpenSearch.")


query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-mpnet-base-v2"))
query_pipeline.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

query = "What are the KPIs for Standard Chartered?"
result = query_pipeline.run({"text_embedder": {"text": query}})

if result['retriever']['documents']:
    answer_document = result['retriever']['documents'][0]
    print(f"Document ID: {answer_document.id}")
    print(f"Content: {answer_document.content}")
    print(f"Metadata: {answer_document.meta}")
else:
    print("No documents found for the query.")

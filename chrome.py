from langchain_community.document_loaders import PyMuPDFLoader
import os
from pathlib import Path

from haystack import Pipeline
from haystack.dataclasses import Document
from haystack.components.writers import DocumentWriter

from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever

# Chroma is used in-memory so we use the same instances in the two pipelines below
document_store = ChromaDocumentStore(persist_path="D:/projects/devposts")

loader = PyMuPDFLoader("D:/projects/devposts/standard-chartered-plc-full-year-2023-report.pdf")
docs = loader.load()
documents = [docs]

chrome_documents = []
for doc in documents:
    for page in doc:
        chrome_documents.append(Document(content=page.page_content, meta=page.metadata))
        
        

indexing = Pipeline()
indexing.add_component("writer", DocumentWriter(document_store))
indexing.run({"writer": {"documents": chrome_documents}})

querying = Pipeline()
querying.add_component("retriever", ChromaQueryTextRetriever(document_store))

query = "what are the KPIs for stanchart?"
results = querying.run({"retriever": {"query": query, "top_k": 2}})

for d in results["retriever"]["documents"]:
    print(f"Content:\n{d.content}\n")
    print(f"Score: {d.score}\n")
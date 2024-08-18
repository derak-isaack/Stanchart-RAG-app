from haystack import Pipeline
from haystack.dataclasses import Document
from haystack.components.writers import DocumentWriter

from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever

import streamlit as st 
from langchain_community.document_loaders import PyMuPDFLoader
import sqlite3 


st.set_page_config("Stanchart RAG application", layout="wide")

document_store = ChromaDocumentStore(persist_path=".")

def connect():
    conn = sqlite3.connect("chroma.sqlite3")
    return conn 
    

querying = Pipeline()
querying.add_component("retriever", ChromaQueryTextRetriever(document_store))

st.title("2023 StanChart Report CHATGPT")
query = st.text_input("Enter your query:")

if query:
    result = querying.run({"retriever": {"query": query, "top_k": 2}})
    
    if result['retriever']['documents']:
        doc = result['retriever']['documents'][0]
        
        with st.container(border=True):
            st.write("**Top Document Result:**")
            st.write(f"**Page Content:** {doc.content}")
            st.write(f"**Metadata:** {doc.meta}")
    else:
        st.write("No results found.")
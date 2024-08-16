from haystack import Pipeline
from haystack_integrations.components.retrievers.opensearch import OpenSearchEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
import streamlit as st
import base64 

st.set_page_config("Standard-Chartered🏦 CHATGPT", layout="wide")

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image = "brave.png"
background_image = get_base64_image(image)

background = f'''
<div style='background-image: url("data:image/png;base64,{background_image}"); 
            background-size: cover; 
            background-repeat: no-repeat; 
            background-attachment: fixed; 
            height: 100vh; 
            width: 100%;'>
</div>
'''

st.html(background)


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


st.title("StanChart CHATGPT")
query = st.text_input("Enter your query:")

if query:
    result = query_pipeline.run({"text_embedder": {"text": query}})
    
    if result['retriever']['documents']:
        doc = result['retriever']['documents'][0]
        
        with st.container(border=True):
            st.write("**Top Document Result:**")
            st.write(f"**Page Content:** {doc.content}")
            st.write(f"**Metadata:** {doc.meta}")
    else:
        st.write("No results found.")

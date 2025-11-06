import streamlit as st
from PyPDF2 import PdfReader
import textwrap
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import google.generativeai as genai

API_KEY = "Your Token Goes Here!"
genai.configure(api_key=API_KEY)

st.title("GenAI common project - 2: RAG based chat bot")
st.sidebar.title("About")

uploaded_files = st.file_uploader("", accept_multiple_files=True)

if uploaded_files: 
    overall_content = ""
    # Step 1: Ingestion (Retrieve and extract text from PDFs)
    for uploaded_file in uploaded_files:
        pdf_file = PdfReader(uploaded_file)
        
        text = ""
        for page in pdf_file.pages:
            text += page.extract_text()
        
        overall_content += text + "\n"

    # Step 2: Chunking (Split text into manageable pieces)
    chunks = textwrap.wrap(overall_content, width=400)

    # Step 3: Embedding (Convert text chunks into vector representations)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(chunks)

    # Step 4: Store embeddings (in Vector Database)
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Step 5: Retrieval from Vector DB based on user query
    def retrieve_similar_chunks(user_query, top_k=3):
        query_embedding = embedding_model.encode([user_query])
        distances, indices = index.search(np.array(query_embedding), top_k)

        retrieved_chunks = [chunks[i] for i in indices[0]]

        return retrieved_chunks 
    
    # Step 6: Generate response using retrieved chunks
    def generate_response(query, context_chunks):
        context = "\n".join(context_chunks)

        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content(prompt)
        
        return response
    
    user_query = st.chat_input("Enter your query here!!")
    
    if user_query: 
        related_info_chunks = retrieve_similar_chunks(user_query)
        answer = generate_response(user_query, related_info_chunks)
        st.markdown(f"Bot: {answer.text}")
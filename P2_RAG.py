import streamlit as st
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
import openai

st.title("Projet 2 : Création d'une architecture Retrieval Augmented Generation sur des descriptions de produits Amazon")

st.write("### Donnée utilisée : metadata.jsonl")
# Chemin du fichier JSONL
uploaded_file = 'meta.jsonl'

# Fonction pour charger le fichier JSONL
def load_jsonl(file):
    # Charger le fichier JSONL dans un DataFrame pandas
    df = pd.read_json(file, lines=True)
    return df

# Charger et afficher les premières lignes du fichier
try:
    df = load_jsonl(uploaded_file)
except Exception as e:
    st.error(f"Erreur lors du chargement du fichier : {e}")

# Diviser les descriptions longues en morceaux de taille appropriée
chunk_size = 512
chunk_overlap = 128

# Vérifier si la colonne 'details' existe dans le DataFrame
if 'details' in df.columns:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    df['details_chunks'] = df['details'].apply(lambda x: text_splitter.split_text(str(x)) if pd.notnull(x) else [])
else:
    st.error("La colonne 'details' n'existe pas dans le fichier JSONL.")

# Charger le modèle d'embedding
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# df['details_chunks'] to list
details_chunks_list = df['details_chunks'].astype(str).tolist()

vectorstore = FAISS.from_texts(
    texts=details_chunks_list,
    embedding=model)

retriever = vectorstore.as_retriever()

# Vérifier la taille du vector store
vector_store_size = vectorstore.index.ntotal
st.write(f"Vector store contains {vector_store_size} vectors.")

# Structurer le prompt pour le LLM
prompt_template = """
You are a helpful assistant designed to answer user queries using only the information extracted from a set of product descriptions. 
Your responses should strictly rely on the provided documents. If you cannot find an appropriate answer in the documents, respond with "I do not know."

Please make sure to include the exact passages or document details from which you derive your answers. 
Do not generate any information beyond what is provided in the data, and avoid sharing sensitive, inappropriate, or potentially incorrect information.

User query: {query}
Relevant documents: {documents}

Your response should be based solely on the relevant documents and passages provided.
"""

# Configuration de l'API OpenAI
openai.api_key = 'your-api-key'


# Créer une fonction pour appeler GPT-3.5 Turbo
def query_gpt3_turbo(query, documents):
    prompt = prompt_template.format(query=query, documents=documents)
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,  # Le modèle doit être plus déterministe
        max_tokens=500,  # Limiter la taille de la réponse
    )
    
    return response.choices[0].message['content']

# Demander à l'utilisateur de saisir une requête
query = st.text_input("Enter your query:", "Products from China?")

if query:
    # Effectuer la récupération des documents pertinents
    results = retriever.get_relevant_documents(query)

    # Afficher les résultats de la récupération
    st.write("Sample retrieval results:")
    # Afficher les résultats de la récupération
    for result in results:
        st.write(result.page_content)  # Utilisez page_content pour accéder au texte du document

    # Préparer les documents pertinents pour GPT-3.5 Turbo
    relevant_documents = "\n".join([result.page_content for result in results])

    # Exécuter GPT-3.5 Turbo avec le prompt structuré
    response = query_gpt3_turbo(query=query, documents=relevant_documents)
    
    # Afficher la réponse générée par GPT-3.5 Turbo
    st.write("Response from GPT-3.5 Turbo:")
    st.write(response)
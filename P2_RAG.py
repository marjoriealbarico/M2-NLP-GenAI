import streamlit as st
import pandas as pd

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
    st.write("Voici les premières lignes du fichier JSONL :")
    st.write(df.head())  # Affiche les 5 premières lignes du DataFrame
except Exception as e:
    st.error(f"Erreur lors du chargement du fichier : {e}")

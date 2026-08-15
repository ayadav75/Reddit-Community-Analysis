import pandas as pd
import torch
import re
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

# Config
INPUT_FILE = "reddit_ALL_data.csv"
OUTPUT_FILE = "reddit_data_with_topics_min_10_auto_reduce_mpnet.csv"
MODEL_PATH = "bertopic_model_saved_new"
"""

6 Parts in bertmodel

1. Embedding documents
2. Reducing dimensionality of embeddings: Locking the random_state to get reproducible results
3. Clustering reduced embeddings into topics
4. Tokenization of topics: I am swapping this with custom tokenizer so that we do not keep common english words like he she me the etc.
5. Weight tokens
6. Represent topics with one or multiple representations: Swapping the default with KeyBERTInspired to get sematic based results rather than just frequency based results. This will help us get more meaningful topics.

"""
def clean_text(text):
    # Basic cleanup: remove URLs and newlines
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+', '', text)
    text = text.replace('\n', ' ').strip()
    return text

def run_modular_modeling():
    print("Step 2.1: Loading Data")
    df = pd.read_csv(INPUT_FILE)
    
    # Filter noise
    df = df.dropna(subset=['combined_text'])
    df['processed_text'] = df['combined_text'].apply(clean_text)
    df = df[df['processed_text'].str.len() > 30]
    docs = df['processed_text'].tolist()
    
    print(f"Processing {len(docs)} documents...")
    

    # Step 1: Embeddings, using all-mpnet-base-v2 for 768 dimensional embeddings
    # I had an rtx 3060 so it got done relatively quickly
    embedding_model = SentenceTransformer('all-mpnet-base-v2', device="cuda")

    # Step 2: Dimensionality Reduction (UMAP)
    # We lock the random_state for reproducible results
    umap_model = UMAP(
        n_neighbors=15, 
        n_components=5, 
        min_dist=0.0, 
        metric='cosine', 
        random_state=42 
    )

    # Step 3: Clustering (HDBSCAN)
    hdbscan_model = HDBSCAN(
        min_cluster_size=10, 
        metric='euclidean', 
        cluster_selection_method='eom', 
        prediction_data=True
    )

    # Step 4: Vectorizer 
    # Removing stopwords
    vectorizer_model = CountVectorizer(
        stop_words="english", 
        ngram_range=(1, 3)  # Allow short phrases, saw good results with this
    )

    # Step 5: c-TF-IDF 
    # No change

    # Step 6: Representation 
    # KeyBERTFocuses on keywords that are semantically similar to the cluster
    representation_model = KeyBERTInspired()

    # Assemble final berttopic model with all the components
    print("Fitting Modular BERTopic")
    topic_model = BERTopic(
        embedding_model=embedding_model,    # Step 1
        umap_model=umap_model,              # Step 2
        hdbscan_model=hdbscan_model,        # Step 3
        vectorizer_model=vectorizer_model,  # Step 4
        representation_model=representation_model, # Step 6
        nr_topics="auto",                   
        min_topic_size=10,                   # Minimum size of topics
        verbose=True
    )

    topics, probs = topic_model.fit_transform(docs)

    # REPORTING
    freq = topic_model.get_topic_info()
    print(f"\nGenerated {len(freq)} topics.")
    print(freq[['Topic', 'Count', 'Name']].head(10))

    # SAVING
    df['Topic'] = topics
    topic_names = {row['Topic']: row['Name'] for _, row in freq.iterrows()}
    df['Topic_Name'] = df['Topic'].map(topic_names)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved results to {OUTPUT_FILE}")
    
    # Save the full model
    topic_model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    run_modular_modeling()
# spinakker
bs
import pandas as pd
import numpy as np
import umap
import hdbscan
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("incident_gaf1_100k.csv", encoding="latin1")
df["short_description"] = df["short_description"].fillna("")
texts = df["short_description"].tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = []
batch_size = 128
for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Texts"):
    batch = texts[i:i + batch_size]
    batch_embeddings = model.encode(batch, convert_to_numpy=True)
    embeddings.append(batch_embeddings)

embeddings = np.vstack(embeddings)

reducer = umap.UMAP(
    n_neighbors=10, 
    n_components=100, 
    min_dist=0.0, 
    metric="cosine", 
    random_state=42
)
reduced_embeddings = reducer.fit_transform(embeddings)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=10, 
    min_samples=1, 
    metric="euclidean", 
    cluster_selection_method="eom",
    prediction_data=True
)
labels = clusterer.fit_predict(reduced_embeddings)

vectorizer = CountVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
text_matrix = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()

cluster_keywords = []
for cluster_id in sorted(set(labels)):
    if cluster_id == -1:
        cluster_keywords.append("Noisy Cluster")
        continue
    cluster_texts = [texts[i] for i in range(len(texts)) if labels[i] == cluster_id]
    cluster_matrix = vectorizer.transform(cluster_texts)
    summed_keywords = np.asarray(cluster_matrix.sum(axis=0)).flatten()
    top_keywords = [feature_names[i] for i in summed_keywords.argsort()[-5:]]
    cluster_keywords.append(", ".join(top_keywords))

df["Cluster"] = labels
df["Cluster Keywords"] = [cluster_keywords[label] if label != -1 else "Noisy Cluster" for label in labels]

import ace_tools as tools
tools.display_dataframe_to_user(name="Original HDBSCAN Topic Modeling", dataframe=df)




import pandas as pd
import numpy as np
import umap
import hdbscan
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("incident_gaf1_100k.csv", encoding="latin1")
df["short_description"] = df["short_description"].fillna("")
texts = df["short_description"].tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = []
batch_size = 256  # Increased batch size for faster embedding generation
for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Texts"):
    batch = texts[i:i + batch_size]
    batch_embeddings = model.encode(batch, convert_to_numpy=True)
    embeddings.append(batch_embeddings)

embeddings = np.vstack(embeddings)

nn = NearestNeighbors(n_neighbors=15, metric="cosine").fit(embeddings)
neighbors = nn.kneighbors(embeddings, return_distance=False)

reducer = umap.UMAP(
    n_neighbors=15, 
    n_components=5,  # Reduced from 100 to 5 for faster clustering
    min_dist=0.0, 
    metric="cosine", 
    random_state=42
)
reduced_embeddings = reducer.fit_transform(embeddings, neighbors=neighbors)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=50,  # Increased for better topic grouping
    min_samples=5,  # Reduced noise points, better cluster cohesion
    metric="euclidean",
    cluster_selection_method="eom",
    core_dist_n_jobs=4
)
labels = clusterer.fit_predict(reduced_embeddings)

vectorizer = CountVectorizer(stop_words="english", ngram_range=(1,2), min_df=5, max_features=10_000)  # More features for better topic coverage
tfidf_transformer = TfidfTransformer()

text_matrix = vectorizer.fit_transform(texts)
tfidf_matrix = tfidf_transformer.fit_transform(text_matrix)
feature_names = vectorizer.get_feature_names_out()

cluster_keywords = []
for cluster_id in sorted(set(labels)):
    if cluster_id == -1:
        cluster_keywords.append("Noisy Cluster")
        continue
    cluster_texts = [texts[i] for i in range(len(texts)) if labels[i] == cluster_id]
    cluster_matrix = vectorizer.transform(cluster_texts)
    tfidf_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    top_keywords = [feature_names[i] for i in tfidf_scores.argsort()[-5:]]
    cluster_keywords.append(", ".join(top_keywords))

df["Cluster"] = labels
df["Cluster Keywords"] = [cluster_keywords[label] if label != -1 else "Noisy Cluster" for label in labels]

import ace_tools as tools
tools.display_dataframe_to_user(name="Optimized HDBSCAN Topic Modeling", dataframe=df)

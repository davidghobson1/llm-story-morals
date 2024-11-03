import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from torch.nn.functional import normalize

from sklearn.preprocessing import minmax_scale
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

from bertopic import BERTopic
from umap import UMAP

import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

DEFAULT_MODEL = "nli-mpnet-base-v2"

# class needed to skip the UMAP step in BERTopic
class NoUMAP(UMAP):
    def fit(self, X):
        return X
    def transform(self, X):
        return X
    
# replace all common names and places with generic ones; text processing for the original full-texts
def replace_common_ents(docs):
    print("Downloading SpaCy to replace the common entities in the texts...")
    import spacy
    import spacy.cli
    spacy.cli.download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")
    replacement_ents = {
        'PERSON': 'Jack', 
        'FAC': 'The Place', 
        'GPE': 'The Place', 
        'LOC': 'The Place'
    }
    print("Replacing common entities... (This may take a few minutes)")
    new_docs = []
    for doc in nlp.pipe(docs):
        new_tokens = [t.text if not t.ent_type_ or not (t.ent_type_ in ['PERSON', 'FAC', 'GPE', 'LOC']) else replacement_ents[t.ent_type_] for t in doc]
        new_docs.append(" ".join(new_tokens))
    return new_docs

def get_optimal_agglomerative_threshold(
        embs, 
        method='complete', 
        lower_bound=0.4, 
        upper_bound=0.9, 
        save_plot=True, 
        output_file="clustering_hyperparameter_plot.jpg"
    ):

    # normalize the embeddings for calinski (it uses euclidean distance; related to cosine distance on unit sphere)
    norm_embs = normalize(embs, dim=1)

    # compute the linkage matrix for agglomerative clustering
    dists = pdist(embs, metric='cosine')
    Z = linkage(dists, method=method)

    # scan through the distance thresholds and compute the Silhouette, Calinksi-Harabasz, and Davies Bouldin scores
    t_vals = np.arange(start=0.2, stop=1, step=0.01).round(2)
    silhouette_scores_agg = []
    calinski_harabasz_scores_agg = []
    davies_bouldin_scores_agg = []
    print("Searching clustering hyperparameter space...")
    for t in tqdm(t_vals):
        clusters = fcluster(Z, t=t, criterion='distance')
        silhouette_scores_agg.append(silhouette_score(embs, clusters, metric='cosine'))
        calinski_harabasz_scores_agg.append(calinski_harabasz_score(norm_embs, clusters))
        davies_bouldin_scores_agg.append(davies_bouldin_score(embs, clusters))
    
    # scale the scores (max -> 1 and min -> 0) and plot the scores
    scaled_silhouette = minmax_scale(silhouette_scores_agg)
    scaled_calinski_harabasz = minmax_scale(calinski_harabasz_scores_agg)
    scaled_product = scaled_silhouette*scaled_calinski_harabasz

    # find the optimal threshold
    lower_cutoff_idx = np.argwhere(t_vals==lower_bound)[0][0]
    upper_cutoff_idx = np.argwhere(t_vals==upper_bound)[0][0]
    optimal_idx = scaled_product[lower_cutoff_idx:upper_cutoff_idx].argmax() + lower_cutoff_idx
    optimal_threshold = t_vals[optimal_idx]

    if save_plot:
        ax = plt.subplot()
        plt.scatter(t_vals, scaled_silhouette, label='Silhouette')
        plt.scatter(t_vals, scaled_calinski_harabasz, label='Calinski-Harabasz')
        plt.scatter(t_vals, scaled_product, label='Product')
        plt.ylabel("Scaled Score")
        plt.xlabel("Distance Threshold")

        plt.legend(prop={'size': 12})
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(15)

        plt.savefig(output_file, bbox_inches='tight')
    
    return optimal_threshold

# fit a BERTopic model to the text; UMAP is skipped since it didn't lead to performance gains for our paper
def get_topic_model(docs, model_name, dist_thres):
    topic_model = BERTopic(
        embedding_model = model_name,
        umap_model = NoUMAP(),
        hdbscan_model = AgglomerativeClustering(n_clusters=None, linkage='complete', distance_threshold=dist_thres, metric='cosine'),
        vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1,1)),
    )
    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform moral (or full-sentence) clustering')
    parser.add_argument('-input_csv', "--input_csv", help="path to the .csv file with the data to be processed. Default is ../data/application/gpt_responses_fairytalez.csv", default="../data/application/gpt_responses_fairytalez.csv")
    parser.add_argument('-col', "--column_name", help="column name of the data to cluster from the .csv. Default is 'moral' (the full-sentence morals). Other options from our paper include 'moral+', 'moral-', or 'text'. The 'text' column includes processing the full-text with the common entity replacement strategy (detailed in our paper) already applied. To apply this entity replacement from scratch, use 'orig_text'", default='moral')
    parser.add_argument("-model", "--model_name", help=f"sentence transformer model to produce the embeddings. These embeddings will be used for the clustering, and any model that can be loaded using SentenceTransformer(model_name) can be used. Default is {DEFAULT_MODEL} (the one used in our paper).", default=DEFAULT_MODEL)
    parser.add_argument("-output_filepath", "--output_filepath", help="output path for the cluster table and cluster visualization. Default is ../outputs", default="../outputs")
    parser.add_argument("-device", "--device", help="device for the torch tensors. Default is 'cpu'", default="cpu")

    args = parser.parse_args()
    input_filepath = args.input_csv
    col = args.column_name
    model_name = args.model_name
    output_filepath = args.output_filepath
    device = args.device

    # make the output directory if it doesn't exist
    if not os.path.exists(output_filepath):
        os.mkdir(output_filepath)

    df = pd.read_csv(input_filepath)
    model = SentenceTransformer(model_name)
    texts = df.loc[df[col].notna(), col].values 
    if col == 'orig_text': 
        # replace all the common entities (people and places) with generic names
        # to avoid this step, use 'text' as a command line argument instead
        texts = replace_common_ents(texts)

    print("Computing embeddings...")
    embs = model.encode(texts, convert_to_tensor=True).to(device)

    # find the optimal distance threshold for clustering
    optimal_threshold = get_optimal_agglomerative_threshold(
                        embs, 
                        save_plot=True, 
                        output_file=os.path.join(output_filepath, f"{col}_clustering_hyperparameter_plot.jpg")
                    )
    print(f"\tOptimal distance threshold = {optimal_threshold}")
    
    # fit the topic model
    print("Fitting topic model...")
    docs = df.loc[df[col].notna(), col].values
    topic_model, topics = get_topic_model(docs, model_name, optimal_threshold)

    # save the topic results
    # topic labels with the input indices
    topic_labels = df['orig_index'].to_frame().join(pd.Series(topics, name=f'{col}_topic'))
    topic_labels.to_csv(os.path.join(output_filepath, f"{col}_topic_assignment.csv"))
    # topic information
    topic_model.get_topic_info().to_csv(os.path.join(output_filepath, f"{col}_topics.csv"), index=False)
    # topic visualization
    fig = topic_model.visualize_topics()
    fig.update_layout(title={'text': f"Intertopic Distance Map ({col})"})
    fig.write_image(os.path.join(output_filepath, f"{col}_topic_visualization.jpg"))
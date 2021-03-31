#!/usr/bin/env python3

import argparse, json, math
import numpy as np
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cluster
from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser(description="The script takes in a file of sentences, and another file with the sentence embeddings, and outputs the clustered sentences.")
    parser.add_argument("--sentence-file", type=str, required=True, help="Path to the file containing all the sentences.")
    parser.add_argument("--embedding-file", type=str, required=True, help="Path to the file containing all the sentence embeddings.")
    parser.add_argument("--outdir",type=str, required=True, help="Directory where the clustered output will be saved to as a JSON file.")
    parser.add_argument("--num-labels", type=int, help="The number of labels to use for clustering. Default: Three times the square root of the number of sentences, rounded down.")
    return parser 

def cluster_TFIDF(sentence_lists, num_labels=None):
    sentences = [s for l in sentence_lists for s in l]
    vectorizer = TfidfVectorizer(ngram_range=(2, 5), analyzer='char_wb').fit(sentences)
    vectors = vectorizer.transform(sentences)
    clustering_model = cluster.AgglomerativeClustering(n_clusters=num_labels if num_labels else 3*int(math.sqrt(len(sentences))))
    # clustering_model = cluster.AffinityPropagation(random_state=0, damping=0.7)
    # clustering_model = cluster.OPTICS(min_samples=2)
    # clustering_model = cluster.KMeans(n_clusters=args.num_labels, random_state=0)

    clusters = clustering_model.fit(vectors.toarray()).labels_
    clustered_lists = group_cluster(sentence_lists, cluster)
    return clustered_lists # [[((index_sentence, index_token), cluster)]]

def group_cluster(sentence_lists, cluster):
    clustered_lists = []
    i = 0
    for index_sentence, l in enumerate(sentence_lists):
        clustered_list = []
        for index_token, s in enumerate(l):
            clustered_list.append(((index_sentence, index_token), clusters[i]))
            i += 1
        clustered_lists.append(clustered_list)
    return clustered_lists

def cluster_BERT(vector_lists, num_labels=None):
    vectors = np.array([s for l in vector_lists for s in l])
    clustering_model = cluster.AgglomerativeClustering(n_clusters=num_labels if num_labels else 3*int(math.sqrt(len(vectors))))
    # clustering_model = cluster.AffinityPropagation(random_state=0, damping=0.7)
    # clustering_model = cluster.OPTICS(min_samples=2)
    # clustering_model = cluster.KMeans(n_clusters=args.num_labels, random_state=0)

    clusters = clustering_model.fit(vectors).labels_
    clustered_lists = group_cluster(vector_lists, cluster)
    return clustered_lists # [[((index_sentence, index_token), cluster)]]

def map_sentences(sent_list, clustered_lists):
    new_list = []
    for essay in clustered_list:
        new = []
        for (index_sentence, index_token), cluster in essay:
            new.append(sent_list[index_sentence][index_token], cluster)
        new_list.append(new)
    return new_list

def get_keywords(clustered_lists, num_keywords=3):
    #sentences_tokenize = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]
    ##sentences_stem = sentences_tokenize
    ##sentences_stem = [[stemmer.stem(token) for token in sentence] for sentence in sentences_tokenize]
    #sentences_vectorize = vectorizer.fit_transform([' '.join(sentence) for sentence in sentences_stem])
    vectorizer = TfidfVectorizer().fit(s for l in clustered_lists for s, _ in l)
    cluster_dict = {}
    for l in clustered_lists:
        for s, c in l:
            cluster_dict.setdefault(c, []).append(s)

    cluster_keywords = {}
    for cluster_id, sl in cluster_dict.items():
        feat_freq = vectorizer.transform(sl).toarray().sum(axis=0).squeeze()
        max_idx = np.argsort(feat_freq)[-num_keywords:][::-1]
        cluster_keywords[cluster_id] = [vectorizer.get_feature_names()[i] for i in max_idx]

    return cluster_keywords

# if __name__=="__main__":
#     args = get_parser().parse_args()
# 
#     with open(args.sentence_file, 'rt') as f:
#         sentences = [line.rstrip('\r\n') for line in f]
# 
#     print(f"{len(sentences)} of sentences read")
#     num_labels = args.num_labels if args.num_labels else 3*int(math.sqrt(len(sentences)))
#     print(f"Clustering to {num_labels} groups")
#     vectors = np.load(args.embedding_file, allow_pickle=True)
#     
#     if issparse(vectors):
#         vectors = vectors.toarray()
# 
#     clustering = cluster_sentences(vectors, num_labels).tolist()
# 
#     dict_list = [{'sentence_id': i, 'sentence': s, 'cluster_id': c} for i, s, c in zip(range(len(sentences)), sentences, clustering)]
#     
#     filename = Path(args.outdir) / ('sentences_clustered' + '.json')
#     with open(filename, 'w') as f:
#         json.dump(dict_list, f, sort_keys=True, indent=4, ensure_ascii=False)
#     print(f"Clustered sentences saved to {filename}")

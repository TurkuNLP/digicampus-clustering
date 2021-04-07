#!/usr/bin/env python3

import argparse, json, math
import numpy as np
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cluster
from sklearn import metrics
from pathlib import Path

def approximate_cluster_count(get_clustering, vectors, cluster_count):
    if cluster_count == 'simple':
        return int(3*math.sqrt(len(vectors)))

    max_score = -1
    gap = 5
    leniency_limit = 4
    leniency = 0
    for n in range(5, len(vectors), gap):
        clustering = get_clustering(n)
        score = metrics.silhouette_score(vectors, clustering)
        if score > max_score:
            max_score = score
            center = n
            leniency = 0
        else:
            leniency += 1
        if leniency > leniency_limit:
            break
    count_range = list(range(max(center - gap, 5), min(center + gap + 1, len(vectors))))
    scores = [metrics.silhouette_score(vectors, get_clustering(n)) for n in count_range]
    return count_range[scores.index(max(scores))]

#    # lower_bound = max(int(math.log(len(vectors))), 2)
#    lower_bound = min(int(2*math.sqrt(len(vectors))), len(vectors))
#    upper_bound = min(int(7*math.sqrt(len(vectors))), len(vectors))
#    scores = []
#    ch_index_scores = []
#    db_scores = []
#    for n in range(lower_bound, upper_bound):
#        clustering = get_clustering(n)
#        scores.append(metrics.silhouette_score(vectors, clustering))
#        ch_index_scores.append(metrics.calinski_harabasz_score(vectors, clustering))
#        db_scores.append(metrics.davies_bouldin_score(vectors, clustering))
#    # scores = [metrics.silhouette_score(vectors, get_clustering(n)) for n in range(lower_bound, upper_bound)]
#    formatted_scores = [f"{s:.2f}" for s in scores]
#    formatted_ch_index_scores = [f"{s:.2f}" for s in ch_index_scores]
#    formatted_db_scores = [f"{s:.2f}" for s in db_scores]
#    print(f"Scores for cluster sizes: {scores}")
#    print(f"Calinski-Harabasz index scores for cluster sizes: {ch_index_scores}")
#    print(f"Davies-Bouldin scores for cluster sizes: {db_scores}")
#    return scores.index(max(scores))

def group_cluster(sentence_lists, vectors, cluster_count, goodness_method, num_labels=None):
    get_clustering = lambda n: cluster.AgglomerativeClustering(n_clusters=n).fit(vectors).labels_
    # clustering_model = cluster.AffinityPropagation(random_state=0, damping=0.7)
    # get_clustering = lambda _: cluster.OPTICS(min_samples=2).fit(vectors).labels_
    # clustering_model = cluster.KMeans(n_clusters=args.num_labels, random_state=0)
    clusters = get_clustering(num_labels if num_labels else approximate_cluster_count(get_clustering, vectors, cluster_count))
    # clusters = get_clustering(0)

    clustered_lists = []
    i = 0
    for index_sentence, l in enumerate(sentence_lists):
        clustered_list = []
        for index_token, s in enumerate(l):
            clustered_list.append(((index_sentence, index_token), clusters[i]))
            i += 1
        clustered_lists.append(clustered_list)

    return clustered_lists, get_goodness(clustered_lists, vectors, clusters, goodness_method)

def cluster_TFIDF(sentence_lists, **args):
    sentences = [s for l in sentence_lists for s in l]
    vectorizer = TfidfVectorizer(ngram_range=(2, 5), analyzer='char_wb').fit(sentences)
    vectors = vectorizer.transform(sentences).toarray()
    return group_cluster(sentence_lists, vectors, **args) # [[((index_sentence, index_token), cluster)]]

def cluster_BERT(vector_lists, **args):
    vectors = np.array([s for l in vector_lists for s in l])
    return group_cluster(vector_lists, vectors, **args) # [[((index_sentence, index_token), cluster)]]

def map_sentences(sent_list, clustered_lists):
    new_list = []
    for essay in clustered_lists:
        new = []
        for (index_sentence, index_token), cluster in essay:
            new.append((sent_list[index_sentence][index_token], cluster))
        new_list.append(new)
    return new_list

def load_stop_words():
    from nltk.corpus import stopwords
    stop_words = stopwords.words('finnish')
    stop_words = stop_words + [',','.','-','esim','voida','myös']
    return stop_words

def get_keywords(clustered_lists, num_keywords=3):
    #sentences_tokenize = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]
    ##sentences_stem = sentences_tokenize
    ##sentences_stem = [[stemmer.stem(token) for token in sentence] for sentence in sentences_tokenize]
    #sentences_vectorize = vectorizer.fit_transform([' '.join(sentence) for sentence in sentences_stem])

    # remove the # in lemmas
    for i, l in enumerate(clustered_lists):
        for j,(s, c) in enumerate(l):
            clustered_lists[i][j] = (s.replace("#",""), c)

    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=load_stop_words()).fit(s for l in clustered_lists for s, _ in l)
    cluster_dict = {}
    for l in clustered_lists:
        for s, c in l:
            cluster_dict.setdefault(c, []).append(s)

    cluster_keywords = {}
    for cluster_id, sl in cluster_dict.items():
        feat_freq = vectorizer.transform(sl).toarray().sum(axis=0).squeeze()
        max_idx = np.argsort(feat_freq)[-num_keywords:][::-1]
        cluster_keywords[cluster_id] = [vectorizer.get_feature_names()[i] for i in max_idx if feat_freq[i]!=0]

    return cluster_keywords

def get_goodness(clustered_lists, vectors, clusters, goodness_method):
    if goodness_method == 'doc-proportion':
        cluster_dict = {}
        for i, l in enumerate(clustered_lists):
            for s, c in l:
                cluster_dict.setdefault(c, set()).add(i)
        return {k: len(v)/len(clustered_lists) for k, v in cluster_dict.items()}

    silhouette_scores = metrics.silhouette_samples(vectors, clusters)
    cluster_dict = {}
    for s, c in zip(silhouette_scores, clusters):
        cluster_dict.setdefault(c, []).append(s)

    return {k: max(sum(v)/len(v), 0) for k, v in cluster_dict.items()}

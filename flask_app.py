from werkzeug.middleware.dispatcher import DispatcherMiddleware
import flask
from flask import Flask
from flask import render_template, request
import os
import glob
import json
import datetime
import html
import re
from doc import Doc, DocCollection, load_doc_collection
import numpy as np

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["KEEP_TRAILING_NEWLINE"] = True
APP_ROOT = os.environ.get('DIGI_CLUSTERING_ROOT',"")
app.config["APPLICATION_ROOT"] = APP_ROOT


DATADIR=os.environ["DIGI_CLUSTERING_DATA"]
exams={} #exam-id -> DocCollection()
for fname in glob.glob(os.path.join(DATADIR,"*.pickle")):
    print("LOADING",fname)
    exam=load_doc_collection(fname)
    exams[exam.id]=exam

@app.route("/")
def index():
    global exams
    #get the documents here
    return render_template("index.html",exams=exams,app_root=APP_ROOT)

@app.route("/<exam_id>")
def exam(exam_id):
    global exams
    exam=exams[exam_id]
    return render_template("exam.html",exam=exam,app_root=APP_ROOT)

def get_clusters_and_kwords(exam,method):
    if method=="TFIDF":
        clusters=exam.TFIDF_clusters
        keywords_lists=exam.TFIDF_keywords
        goodness=exam.TFIDF_goodness
    elif method=="BERT":
        clusters=exam.BERT_clusters
        keywords_lists=exam.BERT_keywords
        goodness=exam.BERT_goodness
    else:
        raise ValueError("Method not recognized")

    keywords={}
    for k,v in keywords_lists.items():
        keywords[k]=", ".join(v)
    return clusters,keywords,goodness

@app.route("/<exam_id>/c/<method>")
def cluster(exam_id,method):
    global exams
    exam=exams[exam_id]
    clusters,keywords,goodness=get_clusters_and_kwords(exam,method)

    cl2sentences={} #clusterid -> [s1,s2,s3,...]
    for clustered_answer in clusters:
        for sent_text,cluster_id in clustered_answer:
            cl2sentences.setdefault(cluster_id,[]).append(sent_text)

    #Again, let's make this easy for ourselves and prep data in python
    cluster_data=[]
    for k in sorted(keywords.keys()):
        cluster_data.append((k,keywords[k],cl2sentences[k][:10],goodness[k]))
    cluster_data.sort(key=lambda c_d:c_d[3],reverse=True)

    return render_template("clusters.html",exam=exam, cluster_data=cluster_data,app_root=APP_ROOT)

@app.route("/<exam_id>/<answer_idx>/<method>/sentence", methods=["POST"])
def get_sentence_cluster(exam_id, answer_idx, method):
    global exams
    exam = exams[exam_id]
    clusters, keywords, goodness = get_clusters_and_kwords(exam, method)
    answer_idx = [d.id for d in exam.docs].index(answer_idx)
    sentence_idx = request.json["sentence_id"]
    sentence_idx = int(sentence_idx.split("_")[-1]) # sentence_idx originally looks like `sentence_12`
    s_text, cls = clusters[answer_idx][int(sentence_idx)] # get the sentence and its cluster

    cl2sentences={} #clusterid -> [s1,s2,s3,...]
    for clustered_answer in clusters:
        for sent_text,cluster_id in clustered_answer:
            cl2sentences.setdefault(cluster_id,[]).append(sent_text)

    #Again, let's make this easy for ourselves and prep data in python
    #cluster_data=(cls, keywords[cls], cl2sentences[cls])
    sentences_html = render_template("sentence.html",
                            cls=cls,
                            keywords=keywords[cls],
                            sentences=cl2sentences[cls],
                            app_root=APP_ROOT)
    return {"sentences_html": sentences_html}

@app.route("/<exam_id>/e/<answer_idx>/<method>")
def answer(exam_id,answer_idx,method):
    answer_idx=int(answer_idx)
    exam=exams[exam_id]
    answer=exam.docs[answer_idx]

    clusters,keywords,goodness=get_clusters_and_kwords(exam,method)


    #Let's perhaps prepare the data for the template here, while we are still in Python
    num_clust=len(keywords)
    clust_sizes=[0 for _ in range(num_clust)]
    for clustered_answer in clusters:
        for sent_text,clust_id in clustered_answer:
            clust_sizes[clust_id]+=1
    clust_sizes=np.array(clust_sizes)

    clust_sizes=(clust_sizes/clust_sizes.sum())
    clust_values=(clust_sizes/clust_sizes.max()*70).astype(np.int) #this is lightness value
    clust_hues=np.arange(0,300,(300-0)/num_clust).astype(np.int)

    ### HERE I PREPARE EVERYTHING I NEED TO VISUALIZE THE TEXT
    ### SO THIS CAN THEN BE LOOPED OVER IN JINJA
    sentences_and_clusters=[]
    for s_text,cls in clusters[answer_idx]:
        br = True if s_text.endswith("\n") else False
        sentences_and_clusters.append((len(sentences_and_clusters),
                                       s_text,
                                       cls,
                                       clust_hues[cls],
                                       clust_values[cls],
                                       br,
                                       goodness[cls]))

    return render_template("answer.html",
                            exam=exam,
                            answer=answer,
                            method=method,
                            sentences_and_clusters=sentences_and_clusters,
                            keywords=keywords,
                            app_root=APP_ROOT)

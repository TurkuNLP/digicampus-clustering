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

@app.route("/e/<exam_id>")
def exam(exam_id):
    global exams
    exam=exams[exam_id]
    return render_template("exam.html",exam=exam,app_root=APP_ROOT)

@app.route("/e/<exam_id>/<answer_idx>/<method>")
def answer(exam_id,answer_idx,method):
    answer_idx=int(answer_idx)
    exam=exams[exam_id]
    answer=exam.docs[answer_idx]

    if method=="TFIDF":
        clusters=exam.TFIDF_clusters
        keywords=exam.TFIDF_keywords

    for k,v in keywords.items():
        keywords[k]=", ".join(v)


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
        sentences_and_clusters.append((s_text,cls,clust_hues[cls],clust_values[cls]))

    return render_template("answer.html",exam=exam,answer=answer,sentences_and_clusters=sentences_and_clusters,keywords=keywords,examapp_root=APP_ROOT)


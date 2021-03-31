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

@app.route("/e/<exam_id>/<answer_id>")
def answer(exam_id,answer_id):
    exam=exams[exam_id]
    answers={}
    for a in exam.docs:
        answers[a.id]=a
    return render_template("answer.html",exam=exam,answer=answers[answer_id],examapp_root=APP_ROOT)


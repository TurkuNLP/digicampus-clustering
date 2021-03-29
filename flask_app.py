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


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
APP_ROOT = os.environ.get('DIGI_CLUSTERING_ROOT',"")
app.config["APPLICATION_ROOT"] = APP_ROOT

DATADIR=os.environ["DIGI_CLUSTERING_DATA"]

@app.route("/")
def index():
    #get the documents here
    exams=[{"id":"example_exam_1"},{"id":"example_exam_2"}]
    return render_template("index.html",exams=exams,app_root=APP_ROOT)


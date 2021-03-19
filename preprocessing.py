#!/usr/bin/env python3

import ufal.udpipe as udpipe

global model, pipeline
model = udpipe.Model.load("fi_model.udpipe")
pipeline = udpipe.Pipeline(model,"tokenize","none","none","horizontal")

def preproc_udpipe(texts):
    texts = [_preproc_udpipe(text) for text in texts]
    return texts

def _preproc_udpipe(text):
    global pipeline
    sent_seg_text = pipeline.process(text)
    return sent_seg_text.split("\n")

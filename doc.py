#!/usr/bin/env python3

import sklearn.feature_extraction
import sklearn.metrics
import numpy
import sys
import os
import json
import pickle
import argparse
#import ufal.udpipe as udpipe
import torch
import transformers
import requests
import tqdm
import pickle
from glob import glob
from read import read_files
from cluster_sentences import cluster_TFIDF, cluster_BERT, get_keywords, get_goodness, map_sentences
import re

def init_models():
    global bert_model, bert_tokenizer
#    global model, pipeline, bert_model, bert_tokenizer
#    assert os.path.exists("fi_model.udpipe"), "You need to download the udpipe model (see readme)"
#    model = udpipe.Model.load("fi_model.udpipe")
#    pipeline = udpipe.Pipeline(model,"tokenize","none","none","horizontal")

    # bert
    bert_model = transformers.BertModel.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")
    bert_model.eval()
    if torch.cuda.is_available():
        bert_model = bert_model.cuda()
    bert_tokenizer = transformers.BertTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")

def embed(data,bert_model,how_to_pool="CLS"):
    with torch.no_grad(): #tell the model not to gather gradients
        mask=data.clone().float() #
        mask[data>0]=1.0
        emb=bert_model(data.cuda(),attention_mask=mask.cuda()) #runs BERT and returns several things, we care about the first
        #emb[0]  # batch x word x embedding
        if how_to_pool=="AVG":
            pooled=emb[0]*(mask.unsqueeze(-1)) #multiply everything by the mask
            pooled=pooled.sum(1)/mask.sum(-1).unsqueeze(-1) #sum and divide by non-zero elements in mask to get masked average
        elif how_to_pool=="CLS":
            pooled=emb[0][:,0,:].squeeze() #Pick the first token as the embedding
        else:
            assert False, "how_to_pool should be CLS or AVG"
            print("Pooled shape:",pooled.shape)
    return pooled.cpu().numpy() #done! move data back to CPU and extract the numpy array

def get_prompt(doc_dict):
    if "topic" in doc_dict:
        prompt = doc_dict["topic"]
    elif "question" in doc_dict:
        prompt = doc_dict["question"]
    else:
        prompt = None
    doc_dict["prompt"] = prompt
    return prompt

class Doc:

    def __init__(self, doc_dict):
        self.doc_dict=doc_dict #this dictionary can have anything the user ever wants but must have "text" field and "id" field
        self.text = doc_dict["essay"]
        self.id = doc_dict["id"]
        self.grade = doc_dict["lab_grade"] if "lab_grade" in doc_dict else None
        if "topic" in doc_dict:
            self.prompt = doc_dict["topic"]
        elif "question" in doc_dict:
            self.prompt = doc_dict["question"]
        else:
            self.prompt = None
        self.lemmas = doc_dict["essay_lemma"] # list of strings, whitespace separation of lemmas
        self.sent_orig = doc_dict["essay_whitespace"] # list of strings, with whitespace preserved
        self.sent_seg_text = doc_dict["sentences"] # list of strings
        #self.preproc_udpipe() # self.sent_seg_text
        self.encode_bert() # self.bert_embedded

    #def preproc_udpipe(self):
    #    global pipeline
    #    sent_seg_text = pipeline.process(self.text)
    #    self.sent_seg_text = [sent.strip() for sent in sent_seg_text.split("\n") if sent.strip()]

    def encode_bert(self):
        tokenized_ids=[bert_tokenizer.encode(txt, add_special_tokens=True) for txt in self.sent_seg_text] #this runs the BERT tokenizer, returns list of lists of integers
        tokenized_ids_t=[torch.tensor(ids,dtype=torch.long) for ids in tokenized_ids] #turn lists of integers into torch tensors
        tokenized_single_batch=torch.nn.utils.rnn.pad_sequence(tokenized_ids_t,batch_first=True)
        self.bert_embedded=embed(tokenized_single_batch,bert_model)
        if len(self.sent_seg_text)==1:
            self.bert_embedded=self.bert_embedded.reshape(1, -1)


class DocCollection:

    def __init__(self, doc_dicts, cluster_count, goodness_method):
        self.docs=[Doc(doc_dict) for doc_dict in tqdm.tqdm(doc_dicts)]
        self.prompt = "\n".join(set([doc.prompt for doc in self.docs]))
        print("Starting clustering...",file=sys.stderr)
        TFIDF_clusters_indices, self.TFIDF_goodness = cluster_TFIDF([doc.sent_seg_text for doc in self.docs], cluster_count=cluster_count, goodness_method=goodness_method)
        self.TFIDF_clusters = map_sentences([doc.sent_orig for doc in self.docs], TFIDF_clusters_indices)
        BERT_clusters_indices, self.BERT_goodness = cluster_BERT([doc.bert_embedded for doc in self.docs], cluster_count=cluster_count, goodness_method=goodness_method)
        self.BERT_clusters = map_sentences([doc.sent_orig for doc in self.docs], BERT_clusters_indices)
        print("Done",file=sys.stderr)
        self.TFIDF_keywords = get_keywords(map_sentences([doc.lemmas for doc in self.docs], TFIDF_clusters_indices))
        self.BERT_keywords = get_keywords(map_sentences([doc.lemmas for doc in self.docs], BERT_clusters_indices))

class CustomUnpickler(pickle.Unpickler):
    """
    https://medium.com/analytics-vidhya/deployment-blues-why-wont-my-flask-web-app-just-deploy-2ac9092a1b40#c18b
    """
    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)

def load_doc_collection(fname):
    collection = CustomUnpickler(open(fname, "rb")).load()
    return collection

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="The script takes in globs for json files and outputs a pickle file of the clustering results.")
    parser.add_argument("--json-glob", type=str, required=True, help="Path to json files containing the essays.")
    parser.add_argument("--out-dir", type=str, required=True, help="Path to the pickle file storing the clustering results.")
    parser.add_argument("--cluster-count", type=str, default='simple', help="What cluster count approximation method to use. Supported values are 'silhouette' and 'simple'. Default: 'simple'.")
    parser.add_argument("--goodness-method", type=str, default='hybrid', help="How to evaluate cluster goodness. Supported values are 'silhouette', 'doc-proportion', and 'hybrid'. Default: 'hybrid'.")
    args = parser.parse_args()

    init_models()
    # example
    files = glob(args.json_glob)
    data = read_files(files)

    all_prompts = list(set([get_prompt(d) for d in data]))
    for prompt in all_prompts:
        prompt_data = [d for d in data if d["prompt"]==prompt]
        if len(prompt_data)>9:
            docs = DocCollection(prompt_data, args.cluster_count, args.goodness_method)
            
            docs.id=prompt[:20].lower().replace("??","a").replace("??","o").replace("??","a").replace(" ","_")
            docs.id=re.sub("[^a-z0-9_]","",docs.id)
            # print(docs.TFIDF_keywords)
            #with open(args.out_pickle,"wb") as f:
            #    pickle.dump(docs, f)
            with open(args.out_dir+"/"+docs.id+".pickle", "wb") as f:
                pickle.dump(docs, f)

    sys.exit(0)

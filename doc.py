#!/usr/bin/env python3

import sklearn.feature_extraction
import sklearn.metrics
import numpy
import sys
import os
import json
import ufal.udpipe as udpipe
import torch
import transformers
import requests
import tqdm
import pickle

from read import read_files
from cluster_sentences import cluster_sentences, get_keywords

def init_models():
    global model, pipeline, bert_model, bert_tokenizer
    assert os.path.exists("fi_model.udpipe"), "You need to download the udpipe model (see readme)"
    model = udpipe.Model.load("fi_model.udpipe")
    pipeline = udpipe.Pipeline(model,"tokenize","none","none","horizontal")

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

class Doc:

    def __init__(self, doc_dict):
        self.doc_dict=doc_dict #this dictionary can have anything the user ever wants but must have "text" field and "id" field
        self.text = doc_dict["essay"]
        self.ID = doc_dict["id"]
        self.preproc_udpipe() # self.sent_seg_text
        self.encode_bert() # self.bert_embedded

    def preproc_udpipe(self):
        global pipeline
        sent_seg_text = pipeline.process(self.text)
        self.sent_seg_text = [sent.strip() for sent in sent_seg_text.split("\n") if sent.strip()]

    def encode_bert(self):
        tokenized_ids=[bert_tokenizer.encode(txt, add_special_tokens=True) for txt in self.sent_seg_text] #this runs the BERT tokenizer, returns list of lists of integers
        tokenized_ids_t=[torch.tensor(ids,dtype=torch.long) for ids in tokenized_ids] #turn lists of integers into torch tensors
        tokenized_single_batch=torch.nn.utils.rnn.pad_sequence(tokenized_ids_t,batch_first=True)
        self.bert_embedded=embed(tokenized_single_batch,bert_model)
        if len(self.sent_seg_text)==1:
            self.bert_embedded=self.bert_embedded.reshape(1, -1)

class DocCollection:

    def __init__(self,doc_dicts):
        self.docs=[Doc(doc_dict) for doc_dict in tqdm.tqdm(doc_dicts)]
        print("Starting clustering...",file=sys.stderr)
        self.TFIDF_clusters = cluster_sentences([doc.sent_seg_text for doc in self.docs])
        print("Done",file=sys.stderr)
        self.TFIDF_keywords = get_keywords(self.TFIDF_clusters)


def make_collection(fnames):
    data = read_files(fnames)
    print("Data jsons read",file=sys.stderr)
    docs = DocCollection(data)
    return docs

    
def main():
    init_models()
    # example
    from glob import glob
    files = glob("tp.json")
    docs=make_collection(files)
    with open("docs.pickle","wb") as f:
        pickle.dump(docs,f)
    print(docs.TFIDF_keywords)
    print(docs.TFIDF_clusters)
    return 0

if __name__=="__main__":
    sys.exit(main())

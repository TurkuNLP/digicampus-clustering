#!/usr/bin/env python3

import sys

from read import read_files
from preprocessing import preproc_udpipe
from cluster_sentences import cluster_sentences, get_keywords


if __name__ == "__main__":
    # read_files(filename_lst) takes a list of json filenames
    filenames = sys.argv[1:]
    data = read_files(filenames) # read json files
    data = [d["essay"] for d in data] # get the essay text from the dictionary item
    data = preproc_udpipe(data) # sentence segmentation by udpipe
    data = cluster_sentences(data)
    keyword_dict = get_keywords(data)

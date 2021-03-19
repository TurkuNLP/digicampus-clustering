#!/usr/bin/env python3

import pickle, sys, os, random, time, math, argparse
random.seed(1)

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import str2bool, get_sentences


def get_parser():
    parser = argparse.ArgumentParser(description="The script takes in a file of sentences, and outputs the sentence embeddings.")
    parser.add_argument("--sentence-file", type=str, required=True, help="Path to the file containing all the sentences.")
    parser.add_argument("--outdir",type=str, required=True, help="Folder where the pickled embedding will be saved to.")
    parser.add_argument("--min-n",type=int, required=True, help="Ngram range, minimum.")
    parser.add_argument("--max-n",type=int, required=True, help="Ngram range, maximum.")
    parser.add_argument("--analyzer",type=str, required=True, help="word, char, or char_wb.")
    parser.add_argument("--stop-words", type=str2bool, nargs='?', const=True, default=False, help="Use stop words or not. Defaults to using.")
    return parser

def encode_with_TFIDF(sentences, nmin, nmax, analyzer, stopwords=True):
    if stopwords:
        stop_words = load_stop_words()
    else:
        stop_words = None
        
    vectorizer = TfidfVectorizer(ngram_range=(nmin,nmax), analyzer=analyzer, stop_words=stop_words)
    sentences_encoded = vectorizer.fit_transform(sentences)  
    ## Tokenization
    #import nltk
    #sentences_tokenize = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]
    ##sentences_stem = sentences_tokenize
    ##sentences_stem = [[stemmer.stem(token) for token in sentence] for sentence in sentences_tokenize]
    #sentences_vectorize = vectorizer.fit_transform([' '.join(sentence) for sentence in sentences_stem])
    return sentences_encoded

def load_stop_words():
    stop_words = stopwords.words('finnish')
    stop_words = stop_words + [',','.','-']
    return stop_words

if __name__=="__main__":
    print('WARNING: This part of the script has not been tested')
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])

    sentences = get_sentences(args.sentence_file)
    print('{} of sentences read'.format(len(sentences)))

    sentences_encoded = encode_with_TFIDF(sentences, args.min_n, args.max_n, args.analyzer, stopwords=args.stop_words)

    filename = args.outdir + '/sentences_encoded_TFIDF_' + str(args.min_n) + str(args.max_n) + args.analyzer + str(args.stop_words) + '.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(sentences_encoded, f)
    print('Encoded sentences saved to {}'.format(filename))

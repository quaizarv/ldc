# Implementation of the "Sentence Similarity Learning by Lexical Decomposition and Composition" Paper
Authors: Zhiguo Wang and Haitao Mi and Abraham Ittycheriah

https://arxiv.org/abs/1602.07019

## Description

Source code for finding Sentence Similarity using the LDC method on WikiQA
Corpus.  The program will take a question and a set of candidate
sentences and assigns a relevance probability to each of them

## Requirements
gensim

## Pretrained Word Vectors

Download pretrained word2vec vectors such as GoogleNews-vectors-negative300.bin
and set embedding type and location of corresponding embedding file in
train_ldc.py. 

> FLAGS.embedding_type = 'GoogleNews'

> FLAGS.w2v_file = '\<w2v-file-path\>/GoogleNews-vectors-negative300.bin'

## Set the location for the WikiQA Corpus and training directory

Download the WikiQACorpus from:

https://www.microsoft.com/en-us/download/confirmation.aspx?id=52419

Update the following parameters in train_ldc.py
> FLAGS.input_dir = '<WikiQACorpus-path>/WikiQACorpus'

> FLAGS.data_dir = '<train-dir-path>/wikiqa-train'

## Training
Set the mode paramter in train_ldc.py to 'train' and then run train_ldc.py
> FLAGS.mode = 'train'

> python train_ldc.py

Takes about 150 epochs (pretty fast on a GPU) to converge

# Testing
Set the mode paramter in train_ldc.py to 'test' and then run train_ldc.py
> FLAGS.mode = 'test'

> python train_ldc.py


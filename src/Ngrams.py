import numpy as np 

def get_count_by_thresh(tokens, thresh):
    vocab = []
    word_counts = {}

    for sent in tokens:
        for word in sent: 
            if word not in word_counts.keys():
                word_counts[word] = 1
            else: 
                word_counts[word] += 1

    for word, cnt in word_counts.items():
        if cnt >= thresh:
            vocab.append(word)

    return vocab

def Ngrams(train_docs, test_docs, n= 3, start_token= '<s>', end_token = '<e>'):
    n_grams = {}
    sentense = [start_token]*n + train_docs + [end_token]
    sentense = tuple(sentense)
    m = len(sentense) - (n-1)
    for i in range(m):
        n_gram = sentense[i:i+n]
        if n_gram in n_grams.keys():
            n_grams[n_gram] += 1
        else: 
            n_grams[n_gram] = 1

    return n_grams

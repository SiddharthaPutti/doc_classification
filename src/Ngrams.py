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


def create_vector_rep(n_gram_freq):
    '''
    creates a matrix of ngrams 
    '''
    
    uniq_keys = set()
    for dictionary in n_gram_freq: 
        uniq_keys.update(dictionary.keys())

    n_rows = len(n_gram_freq)
    n_cols = len(uniq_keys)
    # vec = [[0]* n_cols for _ in range(n_rows)]
    vec = np.zeros((n_rows, n_cols))
    key_indices = {key: index for index, key in enumerate(uniq_keys)}
    for i, dictionary in enumerate(n_gram_freq):
        for key, value in dictionary.items():
            j = key_indices[key]
            vec[i][j] = value

    return vec


def NGrams(train_docs, test_docs, n= 3, start_token= '<s>', end_token = '<e>'):
    '''
    creates ngram for one record at a time 
    '''
    n_grams = {}
    train_list = []
    test_list = []
    print("making Ngrams...")
    for sentense in train_docs:
        local_n_gram = {}
        sentense = [start_token]*n + sentense + [end_token]
        sentense = tuple(sentense)
        m = len(sentense) - (n-1)
        for i in range(m):
            n_gram = sentense[i:i+n]
            if n_gram in n_grams.keys():
                n_grams[n_gram] += 1
            else: 
                n_grams[n_gram] = 1

            if n_gram in local_n_gram.keys():
                local_n_gram[n_gram] +=1
            else: 
                local_n_gram[n_gram] =1

        train_list.append(local_n_gram)
    

    for sentense_test in test_docs:
        sentense_test = [start_token]*n + sentense_test + [end_token]
        sentense_test = tuple(sentense_test)
        p = len(sentense_test) - (n-1)
        n_grams_test = {}
        for i in range(p):
            n_gram = sentense_test[i:i+n]
            if n_gram in n_grams.keys():
                if n_gram in n_grams_test.keys():
                    n_grams_test[n_gram] += 1
                else: 
                    n_grams_test[n_gram] = 1
        test_list.append(n_grams_test)
    print("creating matix...")
    train_vec = create_vector_rep(train_list)
    test_vec = create_vector_rep(test_list)

    return train_vec, test_vec

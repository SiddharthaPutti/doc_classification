import os 
import string
from loaddata import load_data
import numpy as np
import nltk
import random
import pickle
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from Ngrams import NGrams
stopwords_en_punct = set(stopwords.words('english')).union(set(punctuation))
wnl = WordNetLemmatizer()
vectorizer = TfidfVectorizer(max_features = 512)
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

# if __name__ == "__main__":
#     data = load_data("C:/Users/sid31/Downloads/main/NLP/document_classification/data/data_bbc")
#     print(len(data))

def remove_punct(input_str):
    translation_table = str.maketrans("","", string.punctuation)
    no_punct = input_str.translate(translation_table)
    return no_punct

def treebank_tags(penntags):
    tags = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try: 
        return tags[penntags[:2]]
    except: 
        return 'n'
    
def lemmatize(context):
    # sentences = nltk.sent_tokenize(context)
    # lem_words = []
    # for sent in sentences:
    return [wnl.lemmatize(word.lower(), pos=treebank_tags(tag)) for word, tag in pos_tag(word_tokenize(context))]
    # return lem_words


# class tfIdf:
#     def __init__(self, docs):
#         self.docs = docs

#     def fit_transform(self):
#         vocab = []

#         word_counts_per_doc = {}
#         term_frq= []
#         for i in range(len(self.docs)):
#             for word in self.docs[0][i]: 
#                 if word not in word_counts_per_doc: 
#                     word_counts_per_doc[word] = 1
#                 else: 
#                     word_counts_per_doc[word] +=1
#             for word in word_counts_per_doc: 
#                 word_counts_per_doc[word] /= len(word_counts_per_doc)
#             term_frq.append(word_counts_per_doc)
#             word_counts_per_doc.clear()
            


#         pass
#     def trandform(self, text_docs):
#         pass

def VectorizeTfIdf(docs, test_docs):
    print("converting to term freq inverse doc freq...")
    docs = [' '.join(doc) for doc in docs]
    
    tfidf_matrix_train = vectorizer.fit_transform(docs)
    with open("tfidfPickel.pkl", 'wb') as tfidf:
        pickle.dump(vectorizer, tfidf)
    test_docs = [' '.join(doc) for doc in test_docs]
    tfidf_matrix_test = vectorizer.transform(test_docs)
    return tfidf_matrix_train.toarray().tolist(),tfidf_matrix_test.toarray().tolist() 


def lemma(data, reload = False): 
    data_file = 'lemma.pkl'
    src_path = "C:/Users/sid31/Downloads/main/NLP/document_classification"
    src_files = os.listdir(src_path)
    if data_file in src_files:
        if reload: 
            for i in range(len(data)):
                data[i][0] = [word for word in lemmatize(remove_punct(data[i][0])) if word not in stopwords_en_punct and not word.isdigit()]
            with open('loaded_data.pkl', 'wb') as pickel_data:
                pickle.dump(data, pickel_data)
            return data 
        else: 
            print("loading from pickel file...")
            with open('lemma.pkl', 'rb') as loaded_data:
                data = pickle.load(loaded_data)
            return data 

    else: 
        for i in range(len(data)):
            data[i][0] = [word for word in lemmatize(remove_punct(data[i][0])) if word not in stopwords_en_punct and not word.isdigit()]
        with open('loaded_data.pkl', 'wb') as pickel_data:
            pickle.dump(data, pickel_data)
        return data 

def preprocess(data_path):
    # filter, tokenize, 
    # further impliment single letter word exception 
    # exception = ['US], if word in exception add it to lemmetised text else lemmatise(word.lower)
    # data = load_data(data_path)
    
    # load_data by default loads data from pickel file, if the pickel file doesnt present in the directory then it will load the data from text files. 
    # if you have new incoming data just replace reload = True, this will reload the data and save it in pickel file 
    # irreseptive of the file present in the directory. 
    
    data = load_data(path=data_path, reload = True)
    '''
    loaded data definition: 
    data = [
    [data, label], 
    [data, label], 
    . 
    .
    .
    .
    .
    ]
    by default all the returned records are shuffled. 
    '''
    
    print("lemmatizing data...")
    data = lemma(data, reload = True)


    documents = [sublist[0] for sublist in data]
    labels = [sublist[1] for sublist in data]

    train_docs = documents[: int(len(documents)*0.85)]
    train_labels = labels[: int(len(documents)*0.85)]
    test_docs = documents[int(len(documents)*0.85): ]
    test_labels = labels[int(len(documents)*0.85): ]

    # tfidf_train, tfidf_test = VectorizeTfIdf(train_docs, test_docs)

    # # onehot_train, one_hot_test = Ngrams(train_docs, test_docs, n =3)

    # print("making a pickle file of tfidf...")
    # with open('tfidfVector.pkl', 'wb') as vec:
    #     pickle.dump((tfidf_train,tfidf_test, train_labels, test_labels), vec)

    train_vec, test_vec  = NGrams(train_docs, test_docs, n =3)
    print(np.array(train_vec).shape)

def test_preprocess(doc):
    data = [word for word in lemmatize(doc) if word not in stopwords_en_punct and not word.isdigit()]
    data = [' '.join(data) if data else '']
    # print(data)
    with open('tfidfPickel.pkl', 'rb') as tfidf:
        vectorizer = pickle.load(tfidf)
    tfidf_matrix = vectorizer.transform(data)

    return tfidf_matrix.toarray().tolist()
    

if __name__ == "__main__":
    preprocess("C:/Users/sid31/Downloads/main/NLP/document_classification/data/data_bbc")
    # d = load_data("C:/Users/sid31/Downloads/main/NLP/document_classification/data/data_bbc")
# def yeild_data(data):
#     pass # yeild in batches 

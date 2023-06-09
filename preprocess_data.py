import os 
import numpy as np
import nltk
import random
import pickle
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
stopwords_en_punct = set(stopwords.words('english')).union(set(punctuation))
wnl = WordNetLemmatizer()
vectorizer = TfidfVectorizer(max_features = 512)
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

def load_data(path):
    data=[]
    topic_names = os.listdir(path)
    for label, name in enumerate(topic_names):
        data_path = os.path.join(path, name)
        file_names = os.listdir(data_path)
        for file_name in file_names:
            file_path = os.path.join(data_path, file_name)
            with open(file_path, 'r') as f:
                content = f.read()
                data.append([content, label])

    random.shuffle(data)
    X = data[:int(len(data)*0.85)]
    X_test = data[int(len(data)*0.85):]


    # X = data[0][:int(len(data[0])*0.7)]
    # X_valid = data[0][int(len(data[0])*0.7):int(len(data[0])*0.85)]
    # X_test = data[0][int(len(data[0])*0.85):]
    # y = data[1][:int(len(data[1])*0.7)]
    # y_valid = data[1][int(len(data[1])*0.7):int(len(data[1])*0.85)]
    # y_test = data[1][int(len(data[1])*0.85):]

    return X, X_test
    

# if __name__ == "__main__":
#     data = load_data("C:/Users/sid31/Downloads/main/NLP/document_classification/data/data_bbc")
#     print(len(data))
def treebank_tags(penntags):
    tags = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try: 
        return tags[penntags[:2]]
    except: 
        return 'n'
    
def lemmatize(context):
    return [wnl.lemmatize(word.lower(), pos=treebank_tags(tag)) for word, tag in pos_tag(word_tokenize(context))]

def VectorizeTfIdf(docs):
    print("converting to term freq inverse doc freq...")
    # docs = [' '.join(doc) for doc in docs]
    
    tfidf_matrix = vectorizer.fit_transform(docs)
    with open("tfidfPickel.pkl", 'wb') as tfidf:
        pickle.dump(vectorizer, tfidf)
    return tfidf_matrix.toarray().tolist()

def preprocess(data_path):
    # filter, tokenize, 
    # further impliment single letter word exception 
    # exception = ['US], if word in exception add it to lemmetised text else lemmatise(word.lower)
    print("loading data...")
    data , data_test= load_data(data_path)
    print("lemmatizing data...")
    for i in range(len(data)):
        data[i][0] = [word for word in lemmatize(data[i][0]) if word not in stopwords_en_punct and not word.isdigit()]
    documents = [sublist[0] for sublist in data]
    labels = [sublist[1] for sublist in data]

    tfidf = VectorizeTfIdf(documents)
    print("making a pickle file of tfidf...")
    with open('tfidfVector.pkl', 'wb') as vec:
        pickle.dump([tfidf, labels, data_test], vec)
    # return [tfidf, labels]

def test_preprocess(doc):
    data = [word for word in lemmatize(doc) if word not in stopwords_en_punct and not word.isdigit()]
    # data = [' '.join(data) for data in data]
    print(data)
    with open('tfidfPickel.pkl', 'rb') as tfidf:
        vectorizer = pickle.load(tfidf)
    tfidf_matrix = vectorizer.transform(data)

    return tfidf_matrix.toarray().tolist()
    

if __name__ == "__main__":
    preprocess("C:/Users/sid31/Downloads/main/NLP/document_classification/data/data_bbc")
# def yeild_data(data):
#     pass # yeild in batches 

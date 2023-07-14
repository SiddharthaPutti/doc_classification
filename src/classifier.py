import pandas as pd
import numpy as np 
import tensorflow as tf
import pickle
import random
# call word_attention def with X, y, X_valid, y_valid as arguments 
from keras import Model
from keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Dense, Layer, Embedding
from keras import backend as K
import numpy as np
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_angles(pos, i, d):
    angles = pos / (10000 ** (2 * (i // 2) / d))
    return angles

def positional_encoding(positions, d):
    """
    @positions - maximum number of positions to encode
    @d encoding size
    """
    # initialize a matrix angle_rads of all the angles 
    angle_rads = get_angles(np.arange(positions)[:, np.newaxis],
                            np.arange(d)[np.newaxis, :],
                            d)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):

    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :] 

def FullyConnected(embedding_dim, fully_connected_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(fully_connected_dim, activation='tanh'), 
        tf.keras.layers.Dense(embedding_dim) 
    ])

class EncoderLayer(tf.keras.layers.Layer):

    
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate=0.3, layernorm_eps=1e-6):
        super(EncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.mha = MultiHeadAttention(num_heads=num_heads,
                                      key_dim=embedding_dim)

        self.ffn = FullyConnected(embedding_dim=embedding_dim,
                                  fully_connected_dim=fully_connected_dim)

        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
     
    def call(self, x, training, mask):
        
        """
        @x is a tensor of input size  (batch_size, seq_len)
        @training is a boolean value for dropout layer to be active while training and not in predicting 
        @mask - takes the dimensions of x, to determine padding is not part of input.
        
        output shape - (batch_size, input_seq_len, embedding_dim)
        
        """
        
        attn_output, attention_scores = self.mha(x, x, x, mask, return_attention_scores = True) 
        attn_output = self.dropout1(attn_output, training)
        out1 = self.layernorm1(x + attn_output) 
        ffn_output = self.ffn(out1) 
        ffn_output = self.dropout2(ffn_output, training)
        out2 = self.layernorm2(out1 + ffn_output)  
        return out2 


class Encoder(tf.keras.layers.Layer):
    
    """
    The encoder architecture starts by adding embedding to the positional encoding of the embedding layer
    """
    
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size,
               maximum_position_encoding, dropout_rate=0.3, layernorm_eps=1e-6):
        super(Encoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = Embedding(input_vocab_size, self.embedding_dim, input_length = 512)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.embedding_dim)


        self.enc_layers = [EncoderLayer(embedding_dim=self.embedding_dim,
                                        num_heads=num_heads,
                                        fully_connected_dim=fully_connected_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps) 
                           for _ in range(self.num_layers)]

        self.dropout = Dropout(dropout_rate)
        
    def call(self, x, training, mask):
        
        """
        @x is a tensor of input size  (batch_size, seq_len)
        @training is a boolean value for dropout layer to be active while training and not in predicting 
        @mask - takes the dimensions of x, to determine padding is not part of input.
        
        output shape - (batch_size, input_seq_len, embedding_dim)
        
        """
        
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  
        
        # original transformer paper section 3.4 - https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
        # not explained by google why do we use sqrt(model)
        # this is not required but It is to make the positional encoding relatively smaller. 
        # This means the original meaning in the embedding vector won’t be lost when we add them together.
        x *= np.sqrt(self.embedding_dim) 
        # adding scaled embedding to positional encoding 
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training)
        
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x 

class Attention_details(Model):
    
    """
    this impliments encoder(self attention) architecture from transformer paper "Attention is all you need"
    followed by a couple of dense layers to learn features from attention architecture. 
    
    outputs - prob of belonging to a class (batch_size, 1)
    
    """
    
    def __init__(self,  num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size,
           maximum_position_encoding):
        super(Attention_details, self).__init__()

        self.encoder = Encoder(num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size,
               maximum_position_encoding)
        self.gbap = tf.keras.layers.GlobalAveragePooling1D()
        self.fc1 = Dense(32, activation = 'tanh')
        self.fc2 = Dense(1, activation = 'sigmoid')
        self.training = True
        
         
    def call(self,inputs):
        x = self.encoder(inputs, self.training, create_padding_mask(inputs))
        x = self.gbap(x)
        
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
def word_attention(X,y, X_valid, y_valid):

    """
    @X - input variable that takes dim (batch_size, seq_len) , for example seq_len = 512 refers to 512 words per each document 
    @y - target varible that takes dim (batch_size, 1), PHI/NON-PHI binary value for each document 
    
    @X_valid - same as @X for validating data 
    @y_valid - same as @y 
    
    """
        
    model = Attention_details(num_layers = 2,                   # number of multiheadattention(mha) layers 
                              embedding_dim = 64, 
                              num_heads = 2,                    # num of heads in mha
                              fully_connected_dim = 128,        # dense dim before residual connections 
                              input_vocab_size = 100000,        # max vocab seq after embedding 
                              maximum_position_encoding = 512)

    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.Precision(), 
                                                                                             tf.keras.metrics.Recall(),
                                                                                             tf.keras.metrics.Accuracy()])


   

    history = model.fit(tf.convert_to_tensor(X, dtype = tf.float32), 
                        tf.convert_to_tensor(y, dtype = tf.float32), 
                        validation_data = (tf.convert_to_tensor(X_valid, dtype = tf.float32),
                                            tf.convert_to_tensor(y_valid, dtype = tf.float32)),
                        batch_size = 8, 
                        epochs =10)
    
    # with open('classifier.pkl', 'wb') as classifier:
    #     pickle.dump(model, classifier)

    model.save('classifier.h5', save_format='tf')


def NaiveBayes(X_train_tfidf, y_train, X_val_tfidf, y_val):
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train_tfidf, y_train)
    y_val_pred = naive_bayes.predict(X_val_tfidf)

    with open('classifier.pkl', 'wb') as classifier: 
        pickle.dump(naive_bayes, classifier)

    accuracy_val = accuracy_score(y_val, y_val_pred)
    print(accuracy_val)




# def make_sratified_sets(X,y):
#     unique_labels = set(list(y))
#     label_samples = {label: [] for label in unique_labels}

#     for feature, label in zip(X, y):
#         label_samples[label].append(feature)

#     for label in unique_labels:
#         random.shuffle(label_samples[label])

#     train_proportion = 0.7  
#     val_proportion = 0.15 
#     test_proportion = 0.15

#     train_samples = {}
#     val_samples = {}
#     test_samples = {}

#     for label in unique_labels:
#         samples = label_samples[label]
#         total_samples = len(samples)
#         train_size = int(total_samples * train_proportion)
#         val_size = int(total_samples * val_proportion)

#         train_samples[label] = samples[:train_size]
#         val_samples[label] = samples[train_size : train_size + val_size]
#         test_samples[label] = samples[train_size + val_size:]

#     # Merge the samples from each label
#     train_set = sum(train_samples.values(), [])
#     train_labels = sum([[label] * len(samples) for label, samples in train_samples.items()], [])
#     val_set = sum(val_samples.values(), [])
#     val_labels = sum([[label] * len(samples) for label, samples in val_samples.items()], [])
#     test_set = sum(test_samples.values(), [])
#     test_labels = sum([[label] * len(samples) for label, samples in test_samples.items()], [])

#     # Shuffle the final sets
#     random.shuffle(train_set)
#     random.shuffle(val_set)
#     random.shuffle(test_set)


if __name__ == "__main__":
    with open('tfidfVector.pkl', 'rb') as file:
        loaded_list = pickle.load(file)
    train_data, test_data, y_train, y_test = loaded_list
    # shuffled = list(zip(X,y))
    # random.shuffle(shuffled)
    # X, y = zip(*shuffled)
    # make_sratified_sets(X,y)
    # print(y)

    # word_attention(np.array(X), y, np.array(X_valid), y_valid)

    X_train, X_valid, y_train, y_valid = train_data[:int(len(train_data)*0.85)] , train_data[int(len(train_data)*0.85):], y_train[:int(len(y_train)*0.85)]  , y_train[int(len(y_train)*0.85):]

    NaiveBayes(np.array(X_train), y_train, np.array(X_valid), y_valid)



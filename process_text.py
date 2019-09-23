import os
import sys
sys.path.append('../')
import numpy as np
from torchtext.vocab import Vectors, GloVe
import pandas as pd
import re
import gensim
from numpy import *
from gensim.models import word2vec


def process(origin_text, glove_vt, vocab, fix_length=50):
    words = origin_text.split()
    words = [word for word in words if word in vocab ]
    if len(words)>fix_length:
        words = words[:50]
    else:
        words = words + [" "]*(50-len(words))
    word_vect = array([glove_vt.get_vecs_by_tokens(x).numpy() for x in words]).ravel()
    return word_vect

def save_file(train_data, test_data, train_label, test_label):
    np.save("./db_pedia_train_data", train_data.values)
    np.save("./db_pedia_test_data", test_data.values)
    np.save("./db_pedia_train_label", train_label.values)
    np.save("./db_pedia_test_label", test_label.values)
    
def generate_db_pedia(path="./"):
    print("start read file")
    db_train_data = pd.read_csv(path+"train1.csv", header=None)
    db_test_data = pd.read_csv(path+"test1.csv", header=None)
    print("read file end")
    db_train_data[3] = db_train_data[1] + db_train_data[2]
    db_test_data[3] = db_test_data[1] + db_test_data[2]
    glove = GloVe(name='6B', dim=50)
    words_set = set(glove.itos)
    print("start process file")
    train_data = db_train_data[3].apply(lambda word:process(word, glove, words_set))
    test_data = db_test_data[3].apply(lambda word:process(word, glove, words_set))
    print("process file end")
    train_label = db_train_data[0].copy()
    test_label = db_test_data[0].copy()
    print("start save file")
    save_file(train_data, test_data, train_label, test_label)
    print("save file end")

generate_db_pedia("./db_pedia/")

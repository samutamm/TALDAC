# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 15:01:56 2018

@author: 3770098
"""
import pandas as pd
import numpy as np
import os
import json
from nltk import sent_tokenize
import sys

def load_data(dir_path):
    dataframes = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".json"): 
            path = os.path.join(dir_path, filename)
            with open(path) as f:    
                data = json.load(f)
            df = pd.DataFrame([x for x in data["Reviews"]])
            df['HotelInfo'] = data['HotelInfo']['HotelID']
            dataframes.append(df)
            
    data = pd.concat(dataframes)
    return data
    
def all_phrases(content):
    phrase_list = []
    print("Extracting sentences")
    N = len(content)
    for i,revue in enumerate(content) :
        if i % 15 == 0:            
            print_progress(int(i/N * 100))
        phrases = sent_tokenize(revue)
        # qc info des ordre des phrases doit etre ajouter
        for j,p in enumerate(phrases):
            phrase_list.append((p, i))
    phrase_list = np.array(phrase_list)
    return phrase_list[:, 0], phrase_list[:, 1]
    
def print_progress(i):
    sys.stdout.write("\r%d%%" % i)
    sys.stdout.flush()
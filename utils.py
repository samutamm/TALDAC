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
    
def all_phrases_of_multiple_documents(content):
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

def all_phrases_of_one_document(revue):
    phrase_list = []
    print("Extracting sentences")
    phrases = sent_tokenize(revue)
    # qc info des ordre des phrases doit etre ajouter
    for j,p in enumerate(phrases):
        phrase_list.append((p, j))
    phrase_list = np.array(phrase_list)
    return phrase_list


def print_progress(i):
    sys.stdout.write("\r%d%%" % i)
    sys.stdout.flush()
    

# load documents
def load_doc(filename):
    file = open(filename, encoding='utf-8')
    text = file.read()
    file.close()
    return text
 
def load_stories_tgz(filename):
    stories = []
   
    tar = tarfile.open(filename, "r:gz")
    for member in tar.getmembers():
        f = tar.extractfile(member)
        if f is not None:
            # split into story and highlights
            story, highlights = split_story(f.read().decode("utf-8") )
            # store
            stories.append({'story':story, 'highlights':highlights})
   
    return stories
 
# split a document into news story and highlights
def split_story(doc):
    # find first highlight
    index = doc.find('@highlight')
    # split into story and highlights
    story, highlights = doc[:index], doc[index:].split('@highlight')
    # strip extra white space around each highlight
    highlights = [h.strip() for h in highlights if len(h) > 0]
    return story, highlights
 
# load all stories in a directory
def load_stories(directory, N=-1):
    stories = list()
    for name in os.listdir(directory):
        filename = directory + '/' + name
        # load document
        doc = load_doc(filename)
        story, highlights = split_story(doc)
        stories.append({'story':story, 'highlights':highlights})
        if(N > 0 and len(stories) >= N):
            break
    dataframes = []
    for example in stories:
        df = pd.DataFrame({'X': example['story'], 'Y': [example['highlights']]})
        dataframes.append(df)
     
    return pd.concat(dataframes)
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:36:41 2018

@author: 3770098
"""


import numpy as np
import pandas as pd
import os
import json
import nltk
from sklearn.metrics.pairwise import cosine_similarity

def load_data(dir_path):
    dataframes = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".json"): 
            path = os.path.join(directory, filename)
            with open(path) as f:    
                data = json.load(f)
           # print([x['Title'], x['ReviewID', 'Author', 'AuthorLocation', 'Ratings', 'Content', 'Date'] for x in data["Reviews"]])
            df = pd.DataFrame([x for x in data["Reviews"]])
            df['HotelInfo'] = data['HotelInfo']['HotelID']
            dataframes.append(df)
            
    data = pd.concat(dataframes)
    print(data.columns)
    return data

directory = "data/"
data = load_data(directory)
phrases1 = all_phrases(data['Content'].loc[0])

def all_phrases(content):
    phrase_list = []
    for revue in content:
        phrases = nltk.sent_tokenize(revue)
        # qc info des ordre des phrases doit etre ajouter
        for p in phrases:
            phrase_list.append(p)
    return phrase_list

def sent_cos(w1,w2):
    words = set()
    for w in w1:
        words.add(w)
    for w in w2:
        words.add(w)
    res = np.zeros((2, len(words)))
    for i,w in enumerate(words):
        res[0][i] = np.count_nonzero(w1 == w)
        res[1][i] = np.count_nonzero(w2 == w)
    return(cosine_similarity(res)[0][1])

def LCSFinder(string1, string2):
    answer = ""
    len1, len2 = len(string1), len(string2)
    for i in range(len1):
        match = ""
        for j in range(len2):
            if (i + j < len1 and string1[i + j] == string2[j]):
                match += string2[j]
            else:
                if (len(match) > len(answer)): answer = match
                match = ""
    return len(answer)



def LCS(s1, s2):
    res = 1 / len(s1)
    res *= sum([max([LCSFinder(word1,word2) for word2 in s2]) for word1 in s1])
    return res
    

def similarite(s1,s2):
    a = 0.9
    w1 = np.array(nltk.word_tokenize(s1))
    w2 = np.array(nltk.word_tokenize(s2))
    return a * sent_cos(w1,w2) + (1 - a) * LCS(w1,w2)

def creer_matrice_adjance(phrases):
    N = len(phrases)    
    matrice = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i == j:
                matrice[i][j] = 1
                continue
            p1 = phrases[i]
            p2 = phrases[j]
            matrice[i][j] = similarite(p1, p2)
    return matrice

def take_paragraphs_until(sents, chars):
    res = ""
    i = 0
    while(len(res) < chars):
        res = res + sents[i] + ". "
        i += 1
    return res

matrice = creer_matrice_adjance(phrases1)

resumes = []
for seuil in np.linspace(0.1, 0.9, 9):
    ranking = [np.count_nonzero(ligne >= seuil) for ligne in matrice]
    df = pd.DataFrame({'phrase': phrases1})
    df['ranking'] = ranking
    ordered = df.sort_values(by='ranking', ascending=False)['phrase']
    resume = take_paragraphs_until(ordered.values, 350)
    resumes.append(resume)
    print(len(resume))
    print(resume)
    print("-----------------------------------")
    
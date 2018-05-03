# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 14:56:38 2018

@author: 3770098
"""

import numpy as np
import pandas as pd
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize

class GraphBasedSummary:
    
    def __init__(self,phrases):
        assert len(phrases) < 400
        self.phrases = phrases
        self.dumping_factor = 0.85
        
    def power_iteration(self,A, num_simulations):
        b_k = np.random.rand(A.shape[0])
        for _ in range(num_simulations):
            b_k1 = np.dot(A, b_k)
            b_k1_norm = np.linalg.norm(b_k1)
            b_k = b_k1 / b_k1_norm
        return b_k
        
    def lex_rank(self, matrix, threshold):
        #res = 1 - self.dumping_factor
        #res += self.dumping_factor * sum([for v in adj_matrix_line])
        matrix[np.where(matrix < threshold)] = 0
        return self.power_iteration(matrix, 100)
        
    def get_ranking(self,matrix, threshold):
        if self.ranking_method=="lexrank":
            return self.lex_rank(matrix, threshold)

        return [np.count_nonzero(ligne >= threshold) for ligne in matrice]
        
    def sent_cos(self, w1,w2):
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
    
    def LCSFinder(self, string1, string2):
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
    
    
    def LCS(self, s1, s2):
        res = 1 / len(s1)
        res *= sum([max([self.LCSFinder(word1,word2) for word2 in s2]) for word1 in s1])
        return res
        
    
    def similarity(self, s1,s2):
        a = 0.9
        w1 = np.array(word_tokenize(s1))
        w2 = np.array(word_tokenize(s2))
        return a * self.sent_cos(w1,w2) + (1 - a) * self.LCS(w1,w2)
    
    def creer_matrice_adjance(self, phrases):
        N = len(phrases)
        print("Create the distance matrix of size : ", N)
        matrice = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    matrice[i][j] = 1
                    continue
                p1 = phrases[i]
                p2 = phrases[j]
                matrice[i][j] = self.similarity(p1, p2)
        return matrice
    
    def take_paragraphs_until(self, df, chars):
        res = []
        res_len = 0
        i = 0
        while(res_len < chars and i < len(df)):
            res.append((df["phrase"].loc[i], df["position"].loc[i]))
            res_len += len(df["phrase"].loc[i]) + 2
            i += 1
        res = np.array(res)
        resume = pd.DataFrame({'phrase': res[:,0], 'position': res[:,1]})
        ordered_resume = resume.sort_values(by='position', ascending=True)['phrase']
        return ". ".join(ordered_resume.values)
    
    def summarize(self, seuil, summary_length=50):
        matrice = self.creer_matrice_adjance(self.phrases[:, 0])
        print("Ranking")
        ranking = self.get_ranking(matrice, seuil)
        
        df = pd.DataFrame({'phrase': self.phrases[:,0], 'position': self.phrases[:,1]})
        df['ranking'] = ranking
        ordered = df.sort_values(by='ranking', ascending=False)#['phrase']
        return ranking, self.take_paragraphs_until(ordered, summary_length)
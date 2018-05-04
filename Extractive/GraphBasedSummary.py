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
        
    def power_iteration(self,N,M, num_simulations):
        p_last = 1/N * np.ones(N)
        p = p_last
        t = 0
        while True:
            t += 1
            p = M.T.dot(p_last)
            delta = np.linalg.norm(p - p_last)
            p_last = p
            if delta < 1e-10 or t > num_simulations:
                break
        return p
        
    def lex_rank(self, matrix, threshold):
        N = matrix.shape[0]
        degree = np.zeros(N)
        for i in range(N):
            for j in range(N):
                if matrix[i,j] > threshold:
                    matrix[i,j] = 1
                    degree[i] += 1
                else:
                    matrix[i,j] = 0
        for i in range(N):
            for j in range(N):
                matrix[i,j] = matrix[i,j] / degree[i]
                
        #matrix[np.where(matrix < threshold)] = 0
        return self.power_iteration(N,matrix, 100)
        
    def get_ranking(self, threshold):
        matrix = self.creer_matrice_adjance(self.phrases[:, 0])
        if self.ranking_method=="lexrank":
            return self.lex_rank(matrix, threshold)

        return [np.count_nonzero(ligne >= threshold) for ligne in matrix]
        
    def sent_cos(self, w1,w2):
        """
        :w1: np.array or words
        :w2: np.array or words
        """
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
        w1 = np.array(word_tokenize(s1.lower()))
        w2 = np.array(word_tokenize(s2.lower()))
        cosinus = self.sent_cos(w1,w2)
        if self.ranking_method == "lexrank":
            return cosinus
        return a * cosinus + (1 - a) * self.LCS(w1,w2)
    
    def creer_matrice_adjance(self, phrases):
        N = len(phrases)
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
    
    def summarize(self, seuil, ranking_method, summary_length=50):
        self.ranking_method = ranking_method
        ranking = self.get_ranking(seuil)
        #print(ranking)
        df = pd.DataFrame({'phrase': self.phrases[:,0], 'position': self.phrases[:,1]})
        df['ranking'] = ranking
        ordered = df.sort_values(by='ranking', ascending=False)#['phrase']
        return self.take_paragraphs_until(ordered, summary_length)
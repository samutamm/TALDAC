# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:36:41 2018

@author: 3770098
"""

import numpy as np
from GraphBasedSummary import GraphBasedSummary
from utils import load_data, all_phrases_of_one_document, load_stories
from metrics import get_rouge_scores
import matplotlib.pyplot as plt


def create_resume(story, threshold = 0.5, summary_length=300):
    summarizer = GraphBasedSummary(all_phrases_of_one_document(story))
    print(threshold)
    return summarizer.summarize(threshold, "lexrank" ,summary_length=summary_length)

stories = load_stories('data/stories/', N=100)
length = stories['X'].apply(len)
stories = stories[np.logical_and(length > 50, 9000 > length)]
stories = stories.sample(10)

stories['summary'] = ""
summary_length = 1000
precisions = []
recalls = []
f_scores = []
s_candidates = [0,0.1,0.2]
for s in s_candidates:
    precisions_tmp = []
    recalls_tmp = []
    f_scores_tmp = []
    for i in range(stories.shape[0]):
        example = stories.iloc[i]
        resume = create_resume(example['X'], threshold=s, summary_length=summary_length)
        reference = ". ".join(example['Y']) + '.'
        r_scores = get_rouge_scores(resume, reference)

        precisions_tmp.append(r_scores['p'])
        recalls_tmp.append(r_scores['r'])
        f_scores_tmp.append(r_scores['f'])

        #stories['summary'].iloc[i] = resume
        #stories['Y'].iloc[i] = reference
    
    print(precisions_tmp)
    print(recalls_tmp)
    print(f_scores_tmp)
    precisions.append(np.mean(precisions_tmp))
    recalls.append(np.mean(recalls_tmp))
    f_scores.append(np.mean(f_scores_tmp))
    
plt.figure()
plt.plot(s_candidates,precisions, label="Precision")
plt.plot(s_candidates,recalls, label="Recall")
plt.plot(s_candidates,f_scores, label="F-score")
plt.legend()
plt.savefig("s_optimal.png")
#stories.to_csv('data/stories_with_summaries.csv')
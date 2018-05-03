# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:36:41 2018

@author: 3770098
"""

import numpy as np
from GraphBasedSummary import GraphBasedSummary
from utils import load_data, all_phrases_of_one_document, load_stories
from PyRouge.PyRouge.pyrouge import Rouge
import matplotlib.pyplot as plt


def create_resume(story, threshold = 0.5, summary_length=300):
    summarizer = GraphBasedSummary(all_phrases_of_one_document(story))
    return summarizer.summarize(threshold, "lexrank" ,summary_length=summary_length)

stories = load_stories('data/stories/', N=100)
length = stories['X'].apply(len)
stories = stories[np.logical_and(length > 50, 9000 > length)]
stories = stories.sample(30)
rouge = Rouge()

stories['summary'] = ""
threshold = 0.5
summary_length = 100
precisions = []
recalls = []
f_scores = []
for s in [0,0.2,0.4,0.6,0.8,0.9]:
    precisions_tmp = []
    recalls_tmp = []
    f_scores_tmp = []
    for i in range(stories.shape[0]):
        example = stories.iloc[i]
        resume = create_resume(example['X'], threshold=threshold, summary_length=summary_length)
        reference = ". ".join(example['Y']) + '.'
        r_scores = rouge.rouge_l([resume], [reference])

        precisions_tmp.append(r_scores[0])
        recalls_tmp.append(r_scores[1])
        f_scores_tmp.append(r_scores[2])

        stories['summary'].iloc[i] = resume
        stories['Y'].iloc[i] = reference
    
    
    precisions.append(np.mean(precisions_tmp))
    recalls.append(np.mean(recalls_tmp))
    f_scores.append(np.mean(f_scores_tmp))
    
plt.figure()
plt.plot(range(len(precisions)),precisions)
plt.plot(range(len(recalls)),recalls)
plt.plot(range(len(f_scores)),f_scores)
plt.savefig("s_optimal.png")
#stories.to_csv('data/stories_with_summaries.csv')
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:36:41 2018

@author: 3770098
"""


import numpy as np
from GraphBasedSummary import GraphBasedSummary
from utils import load_data, all_phrases_of_one_document, load_stories

def create_resume(story, threshold = 0.5, summary_length=300):
    summarizer = GraphBasedSummary(all_phrases_of_one_document(story))
    return summarizer.summarize(threshold, summary_length=summary_length)

stories = load_stories('data/stories/', N=100)
length = stories['X'].apply(len)
stories = stories[np.logical_and(length > 50, 9000 > length)]

stories['summary'] = ""
for i in range(stories.shape[0]):
    story = stories['X'].iloc[i] 
    resume = create_resume(story)
    stories['summary'].iloc[i] = resume
    stories['Y'].iloc[i] = ". ".join(stories['Y'].iloc[i])

stories.to_csv('data/stories_with_summaries.csv')
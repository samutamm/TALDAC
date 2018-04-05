# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:36:41 2018

@author: 3770098
"""


import numpy as np
from GraphBasedSummary import GraphBasedSummary
from utils import load_data, all_phrases_of_one_document, load_stories


#directory = "data/"
#data = load_data(directory)
#hotels = np.unique(data["HotelInfo"])
#document = data[data["HotelInfo"] == hotels[0]].sample(10)
stories = load_stories('data/stories/', N=1)
story = stories['X'].iloc[0]

summarizer = GraphBasedSummary(all_phrases_of_one_document(story))
resume = summarizer.summarize(0.5, summary_length=300)
print(resume)
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:36:41 2018

@author: 3770098
"""


import numpy as np
from GraphBasedSummary import GraphBasedSummary
from utils import load_data, all_phrases


directory = "data/"
data = load_data(directory)
hotels = np.unique(data["HotelInfo"])
document = data[data["HotelInfo"] == hotels[0]].sample(10)


summarizer = GraphBasedSummary(all_phrases(document["Content"]))
resume = summarizer.summarize(0.5, summary_length=300)
print(resume)
#resumes = []
#for seuil in np.linspace(0.1, 0.9, 9):
#    resume = summarizer.summarize(seuil, summary_length=100)
#    print(len(resume))
#    print(resume)
#    print("-----------------------------------")
#    
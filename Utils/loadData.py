import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tarfile

def del_specialChars(text):
    preprocess = text.replace('\n', ' ')
    preprocess = preprocess.replace('--', '')
    preprocess = preprocess.replace(',', '')
    preprocess = preprocess.replace('.', '')
    preprocess = preprocess.replace('"', '')
    preprocess = preprocess.replace('(cnn)', '')
    preprocess = preprocess.replace('(CNN)', '')
    preprocess = preprocess.replace('(', '')
    preprocess = preprocess.replace(')', '')
    return preprocess.replace('\'', ' ') 

"""
def convertWordsToIndex(text, dicWords, nbWords):
    #nbWords is the number of words we want in the summary
    #text must be a dictionnary with stories and highlights
    story = []

    text_x = del_specialChars(text)
    for word in text_x.lower().split():
        if(word in dicWords):
            story.append(dicWords[word])

    return story[:nbWords]
"""

def convertWordsToIndex(text, dicWords):
    #nbWords is the number of words we want in the summary
    #text must be a dictionnary with stories and highlights
    story = []

    text_x = del_specialChars(text)
    for word in text_x.lower().split():
        if(word in dicWords):
            story.append(dicWords[word])

    return story

def convertIndexToWords(textIdx, idx2words):
    #textIdx is an array of idx
    story = []
    for idx in textIdx:
        story.append(idx2words[idx])
            
    return ' '.join(story)

# load documents 
def load_doc(filename):
    # open the file as read only
    file = open(filename, encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
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
def load_stories(directory):
    stories = list()
    for name in os.listdir(directory):
        filename = directory + '/' + name
        # load document
        doc = load_doc(filename)
        # split into story and highlights
        story, highlights = split_story(doc)
        # store
        stories.append({'story':story, 'highlights':highlights})
    return stories

def split_articles(story, highlights, nbPartition):
    sentences = story.split(".")
    nbSentences = len(sentences)
    sizeOfAPartition = nbSentences // nbPartition
    
    newHighlights = "_start_ " + ' '.join(highlights) 
    
    storiesPartitionned = []
    
    if(sizeOfAPartition != 0):
        for i in range(0, nbSentences, sizeOfAPartition):
            newStory = ' '.join(sentences[i:i+sizeOfAPartition])
            newStory = "_start_ " + newStory

            storiesPartitionned.append((newStory, newHighlights))
        return storiesPartitionned
    else:
        return [(story, ' '.join(highlights))]
    

















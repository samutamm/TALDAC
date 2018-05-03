
# coding: utf-8

# In[14]:


import importlib
import loadData as ld
importlib.reload(ld)
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import collections
import sklearn.decomposition as sk


# In[15]:


#Commande pour lancer sur Floyd  sudo floyd run --env keras --mode jupyter --cpu --data atarax/datasets/cnn_storiestgz/1:/data

#directory = "data/CNN"
#stories = ld.load_stories(directory)[:10000]

#Pour le l'entrainement sur le cloud
tarDirectory = "../data/cnn_stories.tgz"
stories = ld.load_stories_tgz(tarDirectory)
len(stories)


# In[16]:


#Embedding pré-entrainé : conceptNet > Glove 
embeddings_index = {}
with open('../data/numberbatch-en.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding


# In[17]:


corpus = ""

for document in stories:
    corpus += ' '.join(ld.del_specialChars(document['story']).split()[:300])
    corpus += ' '.join(ld.del_specialChars(' '.join(document['highlights'])).split()[:300])

corpus = corpus.lower()
words = corpus.split(' ')

allWords = words
words=list(set(words))


VOCAB_SIZE = len(set(words))

if(VOCAB_SIZE > 40000):
    VOCAB_SIZE = 40000
    words = list(np.array(collections.Counter(allWords).most_common(VOCAB_SIZE))[:, 0])
       


# In[18]:


#Recupération des embeddings des mots utilisés 

START_EMBEDDING_DIM = 300
EMBEDDING_DIM = 100

word_embedding_matrix = np.zeros((VOCAB_SIZE, START_EMBEDDING_DIM), 
                                 dtype=np.float32)

dicWords = collections.OrderedDict(zip(words, range(0, VOCAB_SIZE)))

for word, i in dicWords.items():
    if word in embeddings_index:
        word_embedding_matrix[i] = embeddings_index[word]
    else:
        # If word not in CN, create a random embedding for it
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, START_EMBEDDING_DIM))
        embeddings_index[word] = new_embedding
        word_embedding_matrix[i] = new_embedding
        
        
pca = sk.PCA(n_components=EMBEDDING_DIM)

word_embedding_matrix = pca.fit_transform(word_embedding_matrix)

np.shape(word_embedding_matrix)


# In[19]:


#convert x_train and y_train words in index and make padding

x_train = []
y_train = []

print("toto : ", len(list(dicWords)))

for story in stories:
    story_x = story['story']
    #Join all the highlights to get a longer y
    highlights_x = ' '.join(story['highlights'])

    x = ld.convertWordsToIndex(story_x, dicWords, 300)
    y = ld.convertWordsToIndex(highlights_x, dicWords, 30)
    
    x_train.append(np.array(x))
    y_train.append(np.array(y))
    
x_train = pad_sequences(x_train, padding='post')
y_train = pad_sequences(y_train, padding='post')

print("shape x_train : ", np.shape(x_train))
print("shape y_train : ", np.shape(y_train))


# In[20]:


VOCAB_SIZE


# In[21]:


#charger le modèle du cloud

from keras.models import load_model
base_model=load_model('../model/model90000.h5')


# In[9]:




#np.shape(word_embedding_matrix)

print(words[2000])

print(dicWords['clients'])

print()
print()
print()
print(stories[0]['story'][:1000])
tmp = ld.convertWordsToIndex(stories[0]['story'][:1000], dicWords, 300)
print()
print(tmp)
print()
print(ld.convertIndexToWords(tmp, words))
print()
len(set(words))


# In[10]:


#Autre test 

#Ajout
doc_length = np.shape(x_train)[1]

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, Bidirectional, BatchNormalization
from keras import optimizers

#arbitrarly set latent dimension for embedding and hidden units
latent_dim = EMBEDDING_DIM

##### Define Model Architecture ######

########################
#### Encoder Model ####
encoder_inputs = Input(shape=(doc_length,), name='Encoder-Input')

# Word embeding for encoder (ex: Issue Body)
x = Embedding(VOCAB_SIZE, 
              EMBEDDING_DIM, 
              name='Body-Word-Embedding', 
              weights=[word_embedding_matrix],
              trainable=False,
              mask_zero=False)(encoder_inputs)

x = BatchNormalization(name='Encoder-Batchnorm-1')(x)

# We do not need the `encoder_output` just the hidden state.
_, state_h = GRU(latent_dim, return_state=True, name='Encoder-GRU')(x)


encoder_model = Model(inputs=encoder_inputs, 
                      outputs=state_h, 
                      name='Encoder-Model')

seq2seq_encoder_out = encoder_model(encoder_inputs)

########################
#### Decoder Model ####
decoder_inputs = Input(shape=(None,), name='Decoder-Input')  # for teacher forcing

# Word Embedding For Decoder (ex: Issue Titles)
dec_emb = Embedding(VOCAB_SIZE, 
                    EMBEDDING_DIM, 
                    name='Decoder-Word-Embedding',
                    weights=[word_embedding_matrix],
                    trainable=False,
                    mask_zero=False)(decoder_inputs)

dec_bn = BatchNormalization(name='Decoder-Batchnorm-1')(dec_emb)

# Set up the decoder, using `decoder_state_input` as initial state.
decoder_gru = GRU(latent_dim, 
                  return_state=True, 
                  return_sequences=True, 
                  name='Decoder-GRU')

decoder_gru_output, _ = decoder_gru(dec_bn, initial_state=seq2seq_encoder_out)
x = BatchNormalization(name='Decoder-Batchnorm-2')(decoder_gru_output)

# Dense layer for prediction
decoder_dense = Dense(VOCAB_SIZE, 
                      activation='softmax', 
                      name='Final-Output-Dense')

decoder_outputs = decoder_dense(x)


# Seq2Seq Model

#seq2seq_decoder_out = decoder_model([decoder_inputs, seq2seq_encoder_out])
seq2seq_Model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=0.001), 
loss='sparse_categorical_crossentropy', metrics=['accuracy'])



# In[11]:


seq2seq_Model.summary()


# In[12]:


# training
batch_size = 128
epochs = 5
history = seq2seq_Model.fit([x_train, y_train], np.expand_dims(y_train, -1),
          batch_size=batch_size,
          epochs=epochs,
          verbose=1, validation_split=0.12)


# In[22]:


import utils as utl

importlib.reload(utl)

seq2seq_inf = utl.Seq2Seq_Inference(encoder_preprocessor=x_train,
                                 decoder_preprocessor=y_train,
                                 seq2seq_model=base_model,
                                 words2idx=dicWords,
                                 idx2words=words)

articleIdx=30

test = list(x_train[articleIdx])

#print(stories[articleIdx])

print("Vrai résumé : ", ld.convertIndexToWords(y_train[articleIdx], words))
print()


seq2seq_inf.demo_model_predictions(np.array(test))


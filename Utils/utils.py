from keras.layers import Input
from keras.models import Model
import numpy as np
import loadData as ld
import heapq

def extract_decoder_model(model):
    """
    Extract the decoder from the original model.
    Inputs:
    """
    # the latent dimension is the same throughout the architecture so we are going to
    # cheat and grab the latent dimension of the embedding because that is the same as 
    # what is output from the decoder
    latent_dim = model.get_layer('Decoder-Word-Embedding').output_shape[-1]

    # Reconstruct the input into the decoder
    decoder_inputs = model.get_layer('Decoder-Input').input
    dec_emb = model.get_layer('Decoder-Word-Embedding')(decoder_inputs)
    dec_bn = model.get_layer('Decoder-Batchnorm-1')(dec_emb)

    # Instead of setting the intial state from the encoder and forgetting about it, 
    # during inference we are not doing teacher forcing, so we will have to have a 
    # feedback loop from predictions back into the GRU, thus we define this input 
    # layer for the state so we can add this capability
    gru_inference_state_input = Input(shape=(latent_dim,), name='hidden_state_input')

    # we need to reuse the weights that is why we are getting this
    # If you inspect the decoder GRU that we created for training, it will take as 
    # input 2 tensors -> (1) is the embedding layer output for the teacher forcing,
    #                     which will now be the last step's prediction, and will be 
    #                      _start_ on the first time step.
    #                    (2) is the state, which we will initialize with the encoder 
    #                    on the first time step, but then grab the state after the 
    #                    first prediction and feed that back in again.
    gru_out, gru_state_out = model.get_layer('Decoder-GRU')([dec_bn, 
                                                             gru_inference_state_input])

    # Reconstruct dense layers
    dec_bn2 = model.get_layer('Decoder-Batchnorm-2')(gru_out)
    dense_out = model.get_layer('Final-Output-Dense')(dec_bn2)
    decoder_model = Model([decoder_inputs, gru_inference_state_input],
                          [dense_out, gru_state_out])
    return decoder_model


def extract_encoder_model(model):
    """
    Extract the encoder from the original Sequence to Sequence Model.
    """
    encoder_model = model.get_layer('Encoder-Model')
    return encoder_model

class Seq2Seq_Inference(object):
    def __init__(self,
                 encoder_preprocessor,
                 decoder_preprocessor,
                 seq2seq_model,
                 idx2words,
                 words2idx):

        self.pp_body = encoder_preprocessor
        self.pp_title = decoder_preprocessor
        self.seq2seq_model = seq2seq_model
        self.encoder_model = extract_encoder_model(seq2seq_model)
        self.decoder_model = extract_decoder_model(seq2seq_model)
        self.default_max_len_title = self.pp_title.shape[1]
        self.nn = None
        self.rec_df = None
        
        self.idx2words = idx2words
        self.words2idx = words2idx

    def generate_issue_title(self,
                             raw_input_text,
                             max_len_title=None):
        """
        Use the seq2seq model to generate a title given the body of an issue.
        Inputs
        ------
        raw_input: str
            The body of the issue text as an input string
        max_len_title: int (optional)
            The maximum length of the title the model will generate
        """
        if max_len_title is None:
            max_len_title = self.default_max_len_title
            
        # get the encoder's features for the decoder
        #toto = np.reshape(np.array(raw_input_text), (len(np.array(raw_input_text)), 1))
        raw_input_text = np.array([raw_input_text])
        
        body_encoding = self.encoder_model.predict(raw_input_text)
        
        state_value = np.array(self.words2idx['_start_']).reshape(1, 1)

        decoded_sentence = []
        stop_condition = False
        while not stop_condition:
            preds, st = self.decoder_model.predict([state_value, body_encoding])

            #Prediction of the next word
            pred_idx = np.argmax(preds)
            
            #Affichage à supprimer
            """
            print()
            print("Nouvelle itération")
            #Taille du vocab
            #print("preds : ", np.shape(preds))
            
            #Taille de l'état renvoyé 
            #print("st : ", np.shape(st))
            
            #Choix du model du mot le plus probable
            print("indice du mot le plus probable : ", np.argmax(preds[0][0]))
            
            #preds_sort = preds[0][0]
            #preds_sort.sort()
            
            #lTmp = list(preds[0][0])[-5:]
            #print("5 plus grandes ", lTmp)
            
            lBestIndices = list(np.argpartition(preds[0][0], -4)[-4:])
            print("5 indices correspondant : ", lBestIndices)
            
            print("5 mots les plus probables : ")
            for i in lBestIndices:
                print(self.idx2words[i])
            
            """
            #Fin affichage
            
            
            # retrieve word from index prediction
            pred_word_str = self.idx2words[pred_idx]

            if len(decoded_sentence) >= 30:
                stop_condition = True
                break
            decoded_sentence.append(pred_word_str)

            # update the decoder for the next word
            body_encoding = st
            state_value = np.array(pred_idx).reshape(1, 1)

        return ' '.join(decoded_sentence)


    def print_example(self, body_text):
        """
        Prints an example of the model's prediction with complete article
        """
        #print("-- EXEMPLE --")
        #print("-- Article original --")
        
        #print(ld.convertIndexToWords(body_text, self.idx2words))

        #print("-- Generation --")
        gen_title = self.generate_issue_title(body_text)
        #print(gen_title)
        return gen_title


    def demo_model_predictions(self, article):
        # Extract body and title from DF
        body_text = article.tolist()
        return self.print_example(body_text=body_text)
import pyarrow.feather as feather

import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import text_to_word_sequence
import re, os

from keras.models import Model

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from tqdm import tqdm

import torch
import torch.nn as nn

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    cleanr = re.compile('<.*?>')

    string = re.sub(r'\d+', '', string)
    string = re.sub(cleanr, '', string)
    string = re.sub("'", '', string)
    string = re.sub(r'\W+', ' ', string)
    string = string.replace('_', '')


    return string.strip().lower()


def load_data(parameters):
    print("Carregando dataset.")
    dataset_file = parameters['dataset_file']
    lang = parameters['lang']
    load_from = parameters['load_from']

    if load_from == 'csv':
        data = pd.read_csv(dataset_file)
        
        review_lang = "{}{}".format('text_', lang)
        data = pd.DataFrame(data[[review_lang,'sentiment']])
        
        data[review_lang] = data[review_lang].apply(lambda x: x.lower())
        data[review_lang] = data[review_lang].apply(lambda x: clean_str(x))
        data[review_lang] = data[review_lang].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

        if lang == 'pt':
            data.to_feather('./dataset/data_pt.ftr')
        else:
            data.to_feather('./dataset/data_en.ftr')
    else:
        # TODO: file exists
        filename_ftr = "./dataset/data_{}.ftr".format(lang)
        data = feather.read_feather(filename_ftr)

    data.set_axis(['text', 'sentiment'], axis='columns', inplace=True)
    print("Dataset carregado.\n")
    return data

def prepare_data(data, parameters):
    print("Preparando dados de treinamento.")
    lang = parameters['lang']
    max_features = parameters['max_features']
    max_sequence_length = parameters['max_sequence_length']
    
    if lang == 'pt':
        word_lang = 'portuguese'
    else:
        word_lang = 'english'
        
    stop_words = set(stopwords.words(word_lang))
    
    text = []
    for row in data['text'].values:
        word_list = text_to_word_sequence(row)
        no_stop_words = [w for w in word_list if not w in stop_words]
        no_stop_words = " ".join(no_stop_words)
        text.append(no_stop_words)

    tokenizer = Tokenizer(num_words=max_features, split=' ')

    tokenizer.fit_on_texts(text)
    X = tokenizer.texts_to_sequences(text)  
    
    X = pad_sequences(X, maxlen=max_sequence_length)

    word_index = tokenizer.word_index
    Y = pd.get_dummies(data['sentiment']).values
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)

    print("Dados separados!\n")
    return X_train, X_test, Y_train, Y_test, word_index, tokenizer

def load_pre_trained_wv(word_index, num_words, word_embedding_dim, max_features = 5000):
    print("Carregando pesos de treinamento anteriores.")
    embeddings_index = {}
    f = open(os.path.join('./word_embedding', 'glove.6B.{0}d.txt'.format(word_embedding_dim)), encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('%s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((num_words, word_embedding_dim))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    print("Pesos carregados!\n")
    return embedding_matrix


def model(parameters):
    print("\nConstruindo modelo...")
    pre_trained_wv = parameters['pre_trained_wv']
    max_features = parameters['max_features']
    max_sequence_length = parameters['max_sequence_length']
    bilstm = parameters['bilstm']
    embed_dim = parameters['embed_dim']
    
    if pre_trained_wv is True:
        print("USE PRE TRAINED")
        num_words = min(max_features, len(word_index) + 1)
        weights_embedding_matrix = load_pre_trained_wv(word_index, num_words, word_embedding_dim, max_features)
        input_shape = (max_sequence_length,)
        model_input = Input(shape=input_shape, name="input", dtype='int32')    
        embedding = Embedding(
            num_words, 
            word_embedding_dim,
            input_length=max_sequence_length, 
            name="embedding", 
            weights=[weights_embedding_matrix], 
            trainable=False)(model_input)
        if bilstm is True:
            lstm = Bidirectional(LSTM(word_embedding_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm"))(embedding)
        else:
            lstm = LSTM(word_embedding_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm")(embedding)

    else:
        input_shape = (max_sequence_length,)
        model_input = Input(shape=input_shape, name="input", dtype='int32')    

        embedding = Embedding(max_features, embed_dim, input_length=max_sequence_length, name="embedding")(model_input)
        
        if bilstm is True:
            lstm = Bidirectional(LSTM(embed_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm"))(embedding)
        else:
            lstm = LSTM(embed_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm")(embedding)
    
    model_output = Dense(2, activation='softmax', name="softmax")(lstm)
    model = Model(inputs=model_input, outputs=model_output)
    print("Feito!")
    
    return model

"""
Funções e métodos para attention, bert, gru
"""

class BERTGRUSentimentB(nn.Module):
    def __init__(self, bert, output_dim):
        super().__init__()
        self.bert = bert
        self.embedding_dim = bert.config.to_dict()['hidden_size']
        self.gru11 = nn.GRU(self.embedding_dim, 512, num_layers=1, batch_first=True)
        self.gru12 = nn.GRU(512, 256, num_layers=1, batch_first=True)
        self.gru13 = nn.GRU(256, 128, num_layers=1, batch_first=True)
        self.gru21 = nn.GRU(self.embedding_dim, 512, num_layers=1, batch_first=True)
        self.gru22 = nn.GRU(512, 256, num_layers=1, batch_first=True)
        self.gru23 = nn.GRU(256, 128, num_layers=1, batch_first=True)

        self.fc = nn.Linear(256, output_dim)
        
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)

        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, text):
        # text = [batch size, sent len]

        with torch.no_grad():
            embedding = self.bert(text)[0]  # embedding = [batch size, sent len, emb dim]

        output1, _ = self.gru11(embedding)
        output1 = self.dropout1(output1)  # output1 = [batch size, sent len, 512]
        
        output1, _ = self.gru12(output1)
        output1 = self.dropout2(output1)  # output1 = [batch size, sent len, 256]
        
        _, hidden1 = self.gru13(output1)  # hidden1 = [1, batch size, 128]

        reversed_embedding = torch.from_numpy(embedding.detach().cpu().numpy()[:, ::-1, :].copy()).to(self.DEVICE)
        
        output2, _ = self.gru21(reversed_embedding)
        output2 = self.dropout1(output2)  # output2 = [batch size, sent len, 512]
        
        output2, _ = self.gru22(output2)
        output2 = self.dropout2(output2)  # output1 = [batch size, sent len, 256]
        
        _, hidden2 = self.gru23(output2)  # hidden2 = [1, batch size, 128]
        
        hidden = self.dropout2(torch.cat((hidden1[-1, :, :], hidden2[-1, :, :]), dim=1))  # hidden = [batch size, 256]

        output = self.fc(hidden)  # output = [batch size, out dim]

        return output

    
class BERTGRUSentimentS(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        
        self.bert = bert
        
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [batch size, sent len]
                
        with torch.no_grad():
            embedded = self.bert(text)[0]
                
        #embedded = [batch size, sent len, emb dim]
        
        _, hidden = self.rnn(embedded)
        
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden)
        
        #output = [batch size, out dim]
        
        return output
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for id, batch in enumerate(iterator):
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for id, batch in enumerate(iterator):

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def predict_sentiment(model, tokenizer, sentence, tokens_tag, max_input_length = 300):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    indexed = [tokens_tag['init_token_idx']] + tokenizer.convert_tokens_to_ids(tokens) + [tokens_tag['eos_token_idx']]
    tensor = torch.LongTensor(indexed).to(tokens_tag['device'])
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
#     print(prediction)
    return prediction.item()


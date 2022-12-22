import json
from random import randint
from time import sleep
import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from channels.generic.websocket import WebsocketConsumer





class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        #Why transpose for layer norm? Find out...
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time)
class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()
        
        self.cnn1 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)
class BidirectionalLSTM(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalLSTM, self).__init__()
        
        self.lstm = nn.LSTM(rnn_dim, hidden_size, num_layers=1, batch_first=batch_first,
                            bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        return x
class SpeechRecognition(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats,
                 rnn_dim, hidden_size, batch_first, n_classes):
        super(SpeechRecognition, self).__init__()
        
        self.hcnn = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=3//2)
        
        self.cnn = ResidualCNN(out_channels, out_channels, kernel, stride, dropout, n_feats)
        self.fc = nn.Linear(out_channels**2, rnn_dim)
        self.lstm = BidirectionalLSTM(rnn_dim, hidden_size, dropout, batch_first)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, rnn_dim),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(rnn_dim, n_classes)
            )
        
    def forward(self, x):
        x = self.hcnn(x)
        x = self.cnn(x)
        #print(x.shape)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)
        #print(x.shape)
        x = self.fc(x)
        x = self.lstm(x)
        x = self.classifier(x)
        #print(x.shape)
        return x

characters_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
            'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
            's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '<SPACE>']

class TextProcessing:
    def __init__(self, characters):
        if characters != None:
            #For transcription recognization
            self.characters = characters
            self.characters_map = dict()
            self.index_characters_map = dict()
            for i, character in enumerate(self.characters):
                self.characters_map[character] = i
                self.index_characters_map[i] = character
        
    def text_to_int(self, text):
        seq = []
        for ch in text:
            seq.append(self.characters_map[ch])
        return seq
    
    def int_to_text(self, seq):
        string = []
        for i in seq:
            string.append(self.index_characters_map[i])
        return ''.join(string)

#Initialize text processing        

def GreedyDecoder(output, blank_label=26, collapse_repeated=True):

    characters_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '<SPACE>']

    textprocessing = TextProcessing(characters_list)

    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    #targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        #targets.append(textprocessing.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(textprocessing.int_to_text(decode))
    return decodes #, targets









class WSConsumer(WebsocketConsumer):
    #def __init__(self):
    #    self.frame_id = list()
    def connect(self):

        self.accept()

        self.torch_model  = SpeechRecognition(in_channels=1,
                          out_channels=64,
                          kernel=3, stride=1, dropout=0.2,
                          n_feats=64,
                          rnn_dim=256,
                          hidden_size=100,
                          batch_first=True,
                          n_classes=27
        ) 

        self.optimizer = optim.Adam(self.torch_model.parameters(), lr=1e-3)

        self.torch_model.load_state_dict( torch.load('third_model.pt',map_location='cpu'))

        self.mfcc_tranform = torchaudio.transforms.MFCC(sample_rate=16000,n_mfcc=64)

    def receive(self, text_data=None, bytes_data=None):
        #print("EVENT TRIGERED")
        data = json.loads(text_data)
        waveform = torch.tensor(list(data['array'].values()))
        print('WAVEFORM SHAPE: ', waveform.shape)
        mfcc = self.mfcc_tranform(waveform)
        print('MFCC SHAPE: ', mfcc.shape)
        input_tensor =  mfcc[None,None,:,:]
        print('INPUT TENSOR SHAPE: ', input_tensor.shape)
        out = self.torch_model(input_tensor)
        print(out)

        out = F.log_softmax(out, dim=2)
        out = out.transpose(0, 1)
        print(out)

        decode = GreedyDecoder(out.transpose(0, 1))

        print(decode)

        self.send(json.dumps({
            'message': 'decode_callback',
            'command': decode,
            }
        ))
        #decoded_preds, decoded_targets = GreedyDecoder(out.transpose(0, 1), targets, targets_len)


        #torchaudio.save('1.wav',waveform,16000)
        #print(bytes_data.decode("utf-8","ignore"))
        #self.frame_id.append(data['frame_id'])
        #print(data['array'])
        #arr_data_sample = np.array(list(data['array'].values()))
        #print(max(arr_data_sample))

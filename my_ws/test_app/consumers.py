import json
from random import randint
from time import sleep
import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import struct

from channels.generic.websocket import WebsocketConsumer




class TextProcessed:
    def __init__(self,characters = None,commands = None):
        if characters != None:
            #For transcription recognize
            self.characters = characters
            self.characters_map = dict()
            self.index_characters_map = dict()
            for i, character in enumerate(self.characters):
                self.characters_map[character] = i
                self.index_characters_map[i] = character
        if commands != None:
            #for classification
            self.commands = commands
            self.commands_dict = dict()
            self.index_commands_dict = dict()
            for i, command in enumerate(self.commands):
                self.commands_dict[command] = i
                self.index_commands_dict[i] = command
                
    def text2int(self, text):
        int_list = list()
        for ch in text:
            int_list.append(self.characters_map[ch])
        return int_list
    
    def int2text(self, int_list):
        ch_list = list()
        for int_ch in int_list:
            ch_list.append(self.index_characters_map[int_ch])
        return ''.join(ch_list)

class NN2DMEL(nn.Module):
    def __init__(self, num_class):
        super(NN2DMEL,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=1)
        self.dropout1 = nn.Dropout(0.3) 
    
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(768, 256)
        self.dropout5 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256,128)
        self.dropout6 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, num_class)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),kernel_size=3)
        x = self.dropout1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)),kernel_size=3)
        x = self.dropout2(x)
        x = F.relu(self.fc1(x.reshape(-1,x.shape[1] * x.shape[2]*x.shape[3])))
        x = self.dropout5(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout6(x)
        
        x = self.fc3(x)
        
        #print(x.shape)
        return x 



class WSConsumer(WebsocketConsumer):
    #def __init__(self):
    #    self.frame_id = list()
    def connect(self):

        self.accept()

        self.commands_list = ['go','stop','forward','down','left','right']
        self.tp = TextProcessed(commands = self.commands_list)

        self.net = NN2DMEL(num_class=6)
        self.net.load_state_dict(torch.load(
                'epoch_194.pth',
                map_location=torch.device('cpu')

            )
        )
        self.net.eval()

        self.mfcc_tranform = torchaudio.transforms.MFCC(sample_rate=16000,n_mfcc=64)
        self.mel_transform  = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)

    def receive(self, text_data=None, bytes_data=None):

        #print("EVENT TRIGERED")
        #data = json.loads(text_data)
        signal_list = [struct.unpack("f",bytes_data[index*4:index*4+4])[0] for index in range(16000)]
        #print(text_data)
        
        waveform = torch.tensor(signal_list)
        print('WAVEFORM SHAPE: ', waveform.shape)

        #mfcc = self.mfcc_transform(waveform)
        #print('MFCC SHAPE: ', mfcc.shape)
        mel = self.mel_transform(waveform)
        input_tensor =  mel[None,None,:,:]
        print('INPUT TENSOR SHAPE: ', input_tensor.shape)
        out = self.net(input_tensor)
        print(out)

        predicted = torch.max(out.data, 1)

        decode = self.tp.index_commands_dict[int(predicted.indices)]
        print(decode)
        self.send(json.dumps({
            'message': 'decode_callback',
            'command': decode,
            }
        ))
        
        '''
        out = F.log_softmax(out, dim=2)
        out = out.transpose(0, 1)
        print(out)

        decode = GreedyDecoder(out.transpose(0, 1))

        print(decode)
        '''


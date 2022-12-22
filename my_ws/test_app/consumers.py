import json
from random import randint
from time import sleep
import numpy as np

from channels.generic.websocket import WebsocketConsumer

class WSConsumer(WebsocketConsumer):
    #def __init__(self):
    #    self.frame_id = list()
    def connect(self):

        self.accept()
        #self.frame_id = list()
        #for i in range(1000):
        #    print(i)
        #    #self.send(json.dumps({'message': randint(1,100)}))
        #    sleep(1)
    def receive(self, text_data=None, bytes_data=None):
        #print("EVENT TRIGERED")
        data = json.loads(text_data)
        print(list(data['array'].values()))
        #print(bytes_data.decode("utf-8","ignore"))
        #self.frame_id.append(data['frame_id'])
        #print(data['array'])
        #arr_data_sample = np.array(list(data['array'].values()))
        #print(max(arr_data_sample))

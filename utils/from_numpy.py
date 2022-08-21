import random

import networkx as nx
import numpy as np

    
# to resolve the missing node problem
class FromNumpy(object):
    def __init__(self, hidden_size, emb_path, **kwargs):
        super(FromNumpy, self).__init__()
        self.hidden_size = hidden_size
        self.emb = np.load(emb_path)
    
    def train(self,):
        return self.emb




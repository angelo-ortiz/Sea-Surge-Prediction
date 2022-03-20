# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Now I am gonna create the Transformer from scratch following the
# tutorial in https://www.youtube.com/watch?v=U0s0f995w14

#Let's first define the main blocks that are required. 
#We start by the attention block 

class SelfAttention(nn.Module):
    def __init__(self,embed_size,heads):
        super(SelfAttention,self).__init__()
        self.embed_size=embed_size
        self.heads=heads
        self.head_dim=embed_size//heads
        assert (self.head_dim*heads==embed_size), "Embed Size musty be divisible by the number of heads"
    
        self.values=nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.keys=nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.queries=nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.fc_out=nn.Linear(self.head_dim*heads,embed_size,bias=False)
    
    
    def forward(self,values,keys,query,mask):
        N=query.shape[0]
        values_len,query_len,key_len=values.shape[1],query.shape[1],keys.shape[1]
        #Now the values are split to send to the different heads of the self attention
        values=values.reshape(N,values_len,self.heads,self.head_dim)
        keys=keys.reshape(N,key_len,self.heads,self.head_dim)
        queries=query.reshape(N,query_len,self.heads,self.head_dim)
        
        #Now, the output of the atttention mechanism, which here is called energy and
        #kind of refflects the weight (relevance) of each entry of the values. 
        
        energy=torch.einsum("nqhd,nkhd->nhqk",[queries,keys])
        #queries.shape=(N,query_len,heads,heads_dim)
        #keys.shape=(N,key_len,heads,head_dim)
        #energy.shape=(N,heads,queries_len,head_dim)
        #einsum properly rearrange the tensor multiplication to preserve the corresponding dimensions
        #notice that here we perform for every head the matrix multiplication 
        #query.keys
            
        #below we add a mask, that applies for the case of the decoder, where 
        #you do not want to have the sequence to look at future values in the sequence
        if mask is not None:
            energy=energy.masked_fill(mask==0,float("-1e20"))
        attention=torch.softmax(energy/self.embed_size**(-1/2),dim=3)
        
        out=torch.einsum("nhql+nlhd->nqhd",[attention,values]).reshape(
            N,query_len,self.heads*self.head_dim
            )
        #attention.shape=(N,heads,query_len,value_len=key_len)
        #values.shape=(N,value_len,heads,heads_dim)
        #out.shape=(N,query_len,heads,heads_dim)
        
        out=self.fc_out(out)
        return out
        

class Transformer_block(nn.Module):
    def __init__(self,embed_size,heads,dropout,forward_expansion):
        super(Transformer_block,self).__init__()
        self.attention=SelfAttention(embed_size,heads)
        self.norm1=nn.LayerNorm(embed_size)
        self.norm2=nn.LayerNorm(embed_size)
        
        #Bellow it is just the feed forward scheme that puts a wider set of neuros just
        #in between two elements with same dimension
        self.feed_forward=nn.Sequential(
            nn.Linear(embed_size,forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(embed_size*forward_expansion,embed_size)
            )
        #droput is used to prevent overfitting, by randomly setting out some of 
        #the connections
        self.droupout=nn.Dropout(dropout)
    def forward(self,value,key,query,mask):
        attention=self.attention(value, key, query,mask)
        x=self.dropout(self.norm1(attention+query))
        forward=self.feed_forward(x)
        out=self.droupout(self.norm2(forward+x))
        return out
    
class Encoder(nn.Module):
    def __init__(self,
            src_size,
            embed_size,
            num_layers, 
            heads, 
            device, 
            forward_expansion,
            dropout, 
            max_length):
        super(Encoder,self).__init__()
        self.embed_size=embed_size
        self.device=device
        self.seq_embedding=nn.Embedding(src_size,embed_size)
        self.postion_embedding=nn.Embedding(max_length,embed_size)
        
        self.layers=nn.ModuleList(
            [Transformer_block(embed_size,heads,droupout=dropout,forward_expansion=forward_expansion)
             for _ in range(num_layers)
             ]
            )
        self.droupout=nn.Dropout(dropout)
    
    def forward(self,x,mask):
        N,seq_length=x.shape
        position=torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
        out=self.dropout(self.seq_embedding(x)+self.position_embedding(position))
        
        for layer in self.layers:
            out=layer(out,out,out,mask)
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self,embed_size,heads,forward_expansion,dropout,device):
        super(DecoderBlock,self).__init__()
        self.norm=nn.LayerNorm(embed_size)
        self.attention=SelfAttention(embed_size,heads)
        self.transformer_block=Transformer_block(embed_size,heads,dropout,forward_expansion)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x,value,key,trg_mask):
        attention=self.attention(x,x,x,trg_mask)
        query=self.dropout(self.norm(x+attention))
        out=self.transformer_block(value,key,query)
        return out
    
class Decoder(nn.Module):
    def __init__(self,trg_size,
    embed_size,
    num_layers, 
    heads, 
    device, 
    forward_expansion,
    dropout, 
    max_length):
        super(Decoder,self)
        self.device=device
        self.seq_embedding=nn.Embedding(trg_size,embed_size)
        self.position_embedding=nn.Embedding(max_length,embed_size)
        self.layers=nn.ModuleList([DecoderBlock(embed_size,heads, forward_expansion,dropout,device)
                                   for _ in range(num_layers)
                                   ])
        self.fc_out=nn.Linear(embed_size,trg_size)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x,enc_out,trg_mask):
        N,seq_length=x.shape
        position=torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
        x=self.dropout(self.Embedding(x)+self.postion_embedding(position))
        for layer in self.layers:
            x=layer(x,enc_out,enc_out,trg_mask)
        out=self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            src_size,
            trg_size, 
            src_pad_idx,
            trg_pad_idx, 
            embed_size=256,
            num_layers=6,
            forward_expansion=4,
            heads=8, 
            dropout=0, 
            device="cuda", 
            max_length=40*41*41+20
            ):
        super(Transformer,self).__init__()
        self.encoder=Encoder(src_size,
                             embed_size,
                             num_layers,
                             heads,
                             device,
                             forward_expansion, 
                             dropout, 
                             max_length
                             )
        self.decoder=Decoder(
            trg_size, 
            embed_size,num_layers,heads,device,forward_expansion,dropout, 
            max_length
            )
        self.src_pad_idx=src_pad_idx
        self.trg_pad_idx=trg_pad_idx
        self.devicxe=device
    def make_trg_mask(self,trg):
        N,trg_len=trg.shape
        trg_mask=torch.tril(torch.ones((trg_len,trg_len))).expand(N,1,trg_len,trg_len )
        return trg_mask.to(self.device)
    def forward(self,src,trg):
        trg_mask=self.make_trg_mask(trg)
        enc_src=self.encoder(src)
        out=self.decoder(trg,enc_src,trg_mask)
        return out

#Now, with the Transformer built, it is just missing: (many of them might be already done in Angelo's code)
    #Define the training process
    #Pass the loss function
    #Pass the data in the appropriate form, which in this case I think is as 1D
    # vector with 41*41*40+20 (depending in how many maps I want to pass and if I want to pass the surge levels)      

#Now I will load the data 

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

X_train = np.load('/Users/lopetegui/Documents/ENS/M2/Second_Semester/Mallat_Course/Work/X_train_surge.npz')
Y_train = pd.read_csv('/Users/lopetegui/Documents/ENS/M2/Second_Semester/Mallat_Course/Work/Y_train_surge.csv')
X_test = np.load('/Users/lopetegui/Documents/ENS/M2/Second_Semester/Mallat_Course/Work/X_test_surge.npz')

t_slp = X_train['t_slp'] / 3600
t_slp_delta = t_slp - t_slp[:, 0].reshape(-1, 1)
np.allclose(np.round(t_slp_delta), np.round(t_slp_delta)[0])


import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models

rel_timestamps = np.arange(24*5+1, step=3)[:-1]
t_slp_normalisation = 1e5

def preprocessing(X, device, slp_mean=None, slp_std=None, t_slp_mean=None, t_slp_std=None,
                 surge_mean=None, surge_std=None):
    def normalised_tensor(array, mean, std):
        return torch.from_numpy((array - mean) / std).to(device)
    
    slp = X['slp'].reshape(-1, 40, 41, 41)
    slp = np.roll(slp, shift=-11, axis=3)
    if slp_mean is None:
        slp_mean = np.mean(slp)
    if slp_std is None:
        slp_std = np.std(slp)
    slp = normalised_tensor(slp, slp_mean, slp_std)
    
    fst_slp = X['t_slp'][:, 0] / 3600
    fst_slp_tmp = fst_slp.reshape(-1, 1)
    
    def rel_surge_time(index):
        t_surge = X[index] / 3600
        return torch.from_numpy(t_surge - fst_slp_tmp).to(device)

    t_surge1_in = rel_surge_time('t_surge1_input')
    t_surge2_in = rel_surge_time('t_surge2_input')
    t_surge1_out = rel_surge_time('t_surge1_output')
    t_surge2_out = rel_surge_time('t_surge2_output')
    
    if t_slp_mean is None:
        t_slp_mean = np.mean(fst_slp)
    if t_slp_std is None:
        t_slp_std = np.std(fst_slp)
    fst_slp = normalised_tensor(fst_slp, t_slp_mean, t_slp_std)
    
    surge1 = X['surge1_input']
    surge2 = X['surge2_input']
    if surge_mean is None or surge_std is None:
        surges = np.concatenate([surge1, surge2], axis=None)
        surge_mean = np.mean(surges)
        surge_std = np.std(surges)
    surge1_in = normalised_tensor(surge1, surge_mean, surge_std)
    surge2_in = normalised_tensor(surge2, surge_mean, surge_std)
    
    return X['id_sequence'], slp, slp_mean, slp_std, \
            fst_slp, t_slp_mean, t_slp_std, \
            t_surge1_in, t_surge2_in, t_surge1_out, t_surge2_out, \
            surge1_in, surge2_in, surge_mean, surge_std

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device

train_id_seq, train_slp, slp_mean, slp_std, \
train_fst_slp, t_slp_mean, t_slp_std, \
train_t_surge1_in, train_t_surge2_in, train_t_surge1_out, train_t_surge2_out, \
train_surge1_in, train_surge2_in, surge_mean, surge_std = preprocessing(X_train, device)


test_id_seq, test_slp, _, _, \
test_fst_slp, _, _, \
test_t_surge1_in, test_t_surge2_in, test_t_surge1_out, test_t_surge2_out, \
test_surge1_in, test_surge2_in, _, _ = preprocessing(X_test, device, slp_mean, slp_std, t_slp_mean, t_slp_std, surge_mean, surge_std)

loss_weights = torch.linspace(1, 0.1, 10, requires_grad=False)
    
def surge_prediction_metric(surge1_true, surge2_true, surge1_pred, surge2_pred):
    surge1_score = torch.matmul(torch.square(surge1_true - surge1_pred), loss_weights).mean(dim=1)
    surge2_score = torch.matmul(torch.square(surge2_true - surge2_pred), loss_weights).mean(dim=1)

    return surge1_score + surge2_score

#Now I will define the structure of the data to pass to the Transformer: 
    #We have to pass the source data, as a tensor of shape (N_batches, len_seq=40*41*41+20)
    #We have to pass the target data, as a tensor of shape (N_batches, len_seq=20)


def train(model, loss_fn, optmiser, epochs):
    train_loss = []
    train_acc = []
    
    for epoch_num in range(epochs):
        model.train()
        running_loss = []

        for batch, (slp, t_surge1_in, surge1_in, t_surge2_in, surge2_in, t_surge1_out, t_surge2_out, surge1_out, surge2_out) in enumerate(train_dataloader):
            slp = slp.to(device)
            # ...
            surge2_out = surge2_out.to(device)
            surge1_out_pred, surge2_out_pred = model(slp, t_surge1_in, surge1_in, t_surge2_in, surge2_in, t_surge1_out, t_surge2_out)
            
            loss = loss_fn(surge1_out, surge2_out, surge1_out_pred, surge2_out_pred)
            running_loss.append(loss.item())
            
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
        epoch_loss = np.mean(running_loss)
        train_loss.append(epoch_loss)
        print(f'Epoch {epoch_num+1:03d} | Loss: {epoch_loss:.4f}')
        
        if epoch_num % 5 == 0:
            test_loss = evaluate(model, loss_fn)
            print(f'\tTest loss: {test_loss:.4f}')
        
    return train_loss

def evaluate(model, loss_fn):
    with torch.no_grad():
        model.eval()
        losses = []
        for slp, t_surge1_in, surge1_in, t_surge2_in, surge2_in, t_surge1_out, t_surge2_out, surge1_out, surge2_out in enumerate(test_dataloader):
            slp = slp.to(device)
            # ...
            surge2_out = surge2_out.to(device)
            surge1_out_pred, surge2_out_pred = model(slp, t_surge1_in, surge1_in, t_surge2_in, surge2_in, t_surge1_out, t_surge2_out)
            
            loss = loss_fn(surge1_out, surge2_out, surge1_out_pred, surge2_out_pred)
            losses.append(loss.item())
        return np.mean(losses)
    
model = Transformer()
optimiser = optim.Adam(model.parameters())
_ = train(model, surge_prediction_metric, optmiser, epochs=100)
    
    
    
    
        
'''
Created on February 2nd, 2024

@author: Yanay Rosen
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi.distributions import ZeroInflatedNegativeBinomial
from sklearn.cluster import KMeans
from scipy.stats import rankdata
import scanpy as sc
import numpy as np


def full_block(in_features, out_features, p_drop=0.05):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.LayerNorm(out_features),
        nn.ReLU(),
        nn.Dropout(p=p_drop),
    )


class UCEDecoderModel(torch.nn.Module):

    def __init__(self, n_genes=5000, layer_sizes=(1024, 1024), 
                 uce_embedding_size=1280, categorical_variable_dim=None, 
                 dropout=0.05):
        '''
            Decoder Model
            A ZINB decoder for a given UCE input and optionally categorical variable.
            -- n_genes : number of decoded genes
            -- layer_sizes : tuple of layer size ints
            -- uce_embedding_size : embedding size of UCE model
            -- categorical_variable_dim : number of categories if any, will be added as 
                                                         one hot embedding to condition
                                                         UCE embedding
            -- dropout : dropout chance
        '''
        super().__init__()
        
        self.dropout = dropout
        self.n_genes = n_genes
        self.categorical_variable_dim = categorical_variable_dim

        
        layers = []
        previous_layer_size = uce_embedding_size
        if self.categorical_variable_dim is not None:
                previous_layer_size = uce_embedding_size + categorical_variable_dim # will be added as one hot
                
        for layer_size in layer_sizes:
            layers.append(full_block(previous_layer_size, layer_size, self.dropout))
            previous_layer_size = layer_size
        self.decoder = nn.Sequential(*layers)
        last_layer_size = previous_layer_size # final NN layer before ZINB decoder.
        
        # ZINB Decoder
        self.px_decoder = nn.Sequential(
            full_block(last_layer_size, self.n_genes, self.dropout),
            nn.Linear(self.n_genes, self.n_genes)
        )
        self.px_dropout_decoder = nn.Sequential(
            full_block(last_layer_size, self.n_genes, self.dropout),
            nn.Linear(self.n_genes, self.n_genes)
        )
        self.px_rs = torch.nn.Parameter(torch.randn(self.n_genes))
        

    def forward(self, uce_embeds, categorical_labels, library_size):
        ''''
            -- uce_embed : uce embedding
            -- categorical_labels : categorical_labels if any, else None
        '''
        if self.categorical_variable_dim is not None:
            decoded = self.decoder(torch.hstack((uce_embeds, F.one_hot(categorical_labels.long(), num_classes=self.categorical_variable_dim))))
        else:
            decoded = self.decoder(uce_embeds)
        
        # modfiy
        px = self.px_decoder(decoded) 
        # distribute the means by cluster
        px_scale_decode = nn.Softmax(-1)(px)
        
        px_drop = self.px_dropout_decoder(decoded)
        px_rate =  library_size.unsqueeze(1) * px_scale_decode
        px_r = torch.exp(self.px_rs)        
        
        return px_rate, px_r, px_drop
    
    
    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout):
        '''https://github.com/scverse/scvi-tools/blob/master/scvi/module/_vae.py'''
        return -ZeroInflatedNegativeBinomial(
                                                mu=px_rate, theta=px_r, zi_logits=px_dropout
                                            ).log_prob(x).sum(dim=-1).mean()
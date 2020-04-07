#!/usr/bin/env python
# coding: utf-8

import torch.nn as nn

def init_weights(m):
    """ initialize weights of fully connected layer
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# autoencoder with hidden units 20, latent, 20
# Encoder
class Encoder_2(nn.Module):
    def __init__(self, num_inputs, code_dim):
        super(Encoder_2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_inputs, 20),
            nn.ReLU(),
            nn.Linear(20, code_dim))
        self.encoder.apply(init_weights)
    def forward(self, x):
        x = self.encoder(x)
        return x

# Decoder
class Decoder_2(nn.Module):
    def __init__(self, num_inputs, code_dim):
        super(Decoder_2, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, 20),
            nn.ReLU(),
            nn.Linear(20, num_inputs),
            nn.Sigmoid())
        self.decoder.apply(init_weights)
    def forward(self, x):
        x = self.decoder(x)
        return x

# Autoencoder
class autoencoder_2(nn.Module):
    def __init__(self, num_inputs, code_dim):
        super(autoencoder_2, self).__init__()
        self.encoder = Encoder_2(num_inputs, code_dim)
        self.decoder = Decoder_2(num_inputs, code_dim)
    def forward(self, x):
        code = self.encoder(x)
        x = self.decoder(code)
        return code, x

# autoencoder with hidden units 8, 4, latent, 4, 8
class Encoder_3(nn.Module):
    def __init__(self, num_inputs, code_dim):
        super(Encoder_3, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_inputs, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, code_dim))
        self.encoder.apply(init_weights)
    def forward(self, x):
        x = self.encoder(x)
        return x

# Decoder
class Decoder_3(nn.Module):
    def __init__(self, num_inputs, code_dim):
        super(Decoder_3, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, num_inputs),
            nn.Sigmoid())
        self.decoder.apply(init_weights)
    def forward(self, x):
        x = self.decoder(x)
        return x

# Autoencoder
class autoencoder_3(nn.Module):
    def __init__(self, num_inputs, code_dim):
        super(autoencoder_3, self).__init__()
        self.encoder = Encoder_3(num_inputs, code_dim)
        self.decoder = Decoder_3(num_inputs, code_dim)
    def forward(self, x):
        code = self.encoder(x)
        x = self.decoder(code)
        return code, x

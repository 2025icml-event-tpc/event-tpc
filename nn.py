from math import ceil
from typing import Sequence, Callable, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans


class MLP(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: Sequence[int],
                 output_dim: int,
                 activator = nn.ReLU,
                 dropout: float = 0.2,
                 initializer: Optional[Callable] = None):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activator = activator
        self.dropout = dropout

        if hidden_dim:
            l_module = [
                nn.Linear(input_dim, hidden_dim[0]),
                self.activator(),
                nn.Dropout(p=dropout),
            ]

            for i in range(1, len(hidden_dim)):
                l_module.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
                l_module.append(self.activator())
                l_module.append(nn.Dropout(p=dropout))


            l_module.append(nn.Linear(hidden_dim[-1], output_dim))
        else:
            l_module = [nn.Linear(input_dim, output_dim)]

        self.nn = nn.Sequential(*l_module)

        if initializer is not None:
            def _init_weights(module: nn.Module):
                if isinstance(module, nn.Linear):
                    initializer(module.weight)
            self.nn.apply(_init_weights)

    def forward(self, x):
        return self.nn(x)

class LSTMEncoder(nn.Module):
    """Receive input x (N, L, D), output latent state z (N, L, E)
    """
    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 1, dropout: float = 0.2, initializer: Optional[Callable] = None):
        super(LSTMEncoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self._lstm = nn.LSTM(
            input_dim, 
            output_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )

        self._fc = nn.Linear(output_dim, output_dim)

        if initializer is not None:
            for layer_params in self._lstm.all_weights:
                for params in layer_params:
                    if len(params.size()) > 1:
                        # only initialize weights, without bias
                        initializer(params)

    @property
    def spec(self):
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout
        }

    def forward(self, x):
        h_rnn, _ = self._lstm(x)
        output = self._fc(h_rnn)
        
        return output


class MLPProfileEncoder(MLP):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: Sequence[int],
                 output_dim: int,
                 activator = nn.ReLU,
                 dropout: float = 0.2,
                 initializer: Optional[Callable] = None):
        super(MLPProfileEncoder, self).__init__(input_dim, hidden_dim, output_dim, activator, dropout, initializer)

class TransformerEncoder(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 d_model: int, 
                 nhead: int, 
                 dim_feedforward: int, 
                 dropout=0.1, 
                 activation=F.relu, 
                 num_layers=1, 
                 batch_first=True, 
                 norm_first=False,
                 pos_embedding=None):
        super(TransformerEncoder, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = dim_feedforward
        self.batch_first = batch_first
        self.pos_embedding = pos_embedding

        self.projector = nn.Linear(
            in_features=input_dim,
            out_features=d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation=activation, 
            batch_first=batch_first, 
            norm_first=norm_first
        )

        self.nn = nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=num_layers
        )

    def forward(self, x: torch.Tensor, causal_mask, padding_mask):
        L = x.shape[1] if x.ndim == 3 and self.batch_first else x.shape[0]
        
        x = self.projector(x)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding(L, enc_dim=self.d_model)
        x = self.nn(x, mask=causal_mask, src_key_padding_mask=padding_mask, is_causal=True)

        return x

class EventTransformerEncoder(nn.Module):
    def __init__(self, 
                 n_events: int,
                 d_model: int, 
                 nhead: int, 
                 dim_feedforward: int, 
                 dropout=0.1, 
                 activation=F.relu, 
                 num_layers=1, 
                 batch_first=True, 
                 norm_first=False,
                 pos_embedding=None):
        super(EventTransformerEncoder, self).__init__()

        self.d_model = d_model

        self.event_embedding = nn.Embedding(n_events, d_model)
        self.pos_embedding = pos_embedding

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation=activation, 
            batch_first=batch_first, 
            norm_first=norm_first
        )

        self.nn = nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=num_layers
        )

    def forward(self, time_inputs, event_inputs, causal_mask, padding_mask):
        x = self.event_embedding(event_inputs)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding(time_inputs, enc_dim=self.d_model)
        x = self.nn(x, mask=causal_mask, src_key_padding_mask=padding_mask, is_causal=True)

        return x


class TemporalSeqTransformerEncoder(EventTransformerEncoder):
    def __init__(self, 
                 n_events: int, 
                 n_profiles: int,
                 d_model: int, 
                 nhead: int, 
                 dim_feedforward: int, 
                 dropout=0.1, 
                 activation=F.relu, 
                 num_layers=1, 
                 batch_first=True, 
                 norm_first=False, 
                 pos_embedding=None):
        super().__init__(n_events, d_model, nhead, dim_feedforward, dropout, activation, num_layers, batch_first, norm_first, pos_embedding)

        self.profile_embedding = nn.Embedding(n_profiles, d_model)

    def forward(self, time_inputs, profile_inputs, event_inputs, causal_mask, padding_mask):
        x = self.event_embedding(event_inputs) + self.profile_embedding(profile_inputs)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding(time_inputs, enc_dim=self.d_model)
        x = self.nn(x, mask=causal_mask, src_key_padding_mask=padding_mask, is_causal=True)

        return x

class MLPSelector(MLP):
    """Receive latent state z (N, L, E), output assignment probabilities pi (N, L, K)
    """
    def __init__(self, 
                 K: int,
                 input_dim: int, 
                 hidden_dim: Sequence[int],
                 activator = nn.ReLU,
                 dropout: float = 0.2, 
                 initializer: Optional[Callable] = None):
        super(MLPSelector, self).__init__(input_dim, hidden_dim, K, activator, dropout, initializer)

        self.K = K

    def forward(self, x, output_probs=False):
        logits = self.nn(x)
        if output_probs:
            return F.softmax(logits, dim=-1)

        return logits


class MLPPredictor(MLP):
    """Receive latent state z (N, L, E) or embedding e (N, L, E), output predictions y_hat or y_bar 
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: Sequence[int],
                 output_dim: int,
                 activator = nn.ReLU,
                 dropout: float = 0.2,
                 initializer: Optional[Callable] = None):
        super(MLPPredictor, self).__init__(input_dim, hidden_dim, output_dim, activator, dropout, initializer)

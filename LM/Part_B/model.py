import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalDorpout(nn.Module):
    """Variational Dropout: same dropout mask across time steps"""
    def __init__(self, dropout=0.5):
        super(VariationalDorpout, self).__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training:
            return x
        # x: (batch_size, seq_len, emb_size)
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout)
        mask = mask.div_(1 - self.dropout)
        mask = mask.expand_as(x)
        return x * mask


class LM_LSTM_VD(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0,
                 out_dropout=0.1, emb_dropout=0.5, n_layers=1):
        super(LM_LSTM_VD, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Variational Dropout
        self.embedding_variational_dropout = VariationalDorpout(dropout=emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True, dropout=out_dropout if n_layers > 1 else 0 )
        # Variational Dropout
        self.output_variational_dropout = VariationalDorpout(dropout=out_dropout)
        
        self.pad_token = pad_index

        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        # Embedding + variational dropout
        emb = self.embedding(input_sequence)  # (B, T, E)
        emb = self.embedding_variational_dropout(emb)  # Apply variational dropout
        # LSTM
        lstm_out, _ = self.lstm(emb)
        # Output + variational dropout
        lstm_out = self.output_variational_dropout(lstm_out)
        output = self.output(lstm_out).permute(0, 2, 1) # (B, C, T)

        return output

class LM_LSTM_WT(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0,
                 out_dropout=0.1, emb_dropout=0.5, n_layers=1):
        super(LM_LSTM_WT, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True, dropout=out_dropout if n_layers > 1 else 0 )
        
        self.pad_token = pad_index

        self.output = nn.Linear(hidden_size, output_size)

        # Weight Tying
        if emb_size == hidden_size:
            self.output.weight = self.embedding.weight
        else:
            print("Weight tying is not possible. Hidden size should be equal to embedding size.")

    def forward(self, input_sequence):
        # Embedding 
        emb = self.embedding(input_sequence)  # (B, T, E)
        # LSTM
        lstm_out, _ = self.lstm(emb)
        # Output
        output = self.output(lstm_out).permute(0, 2, 1) # (B, C, T)

        return output

import torch
import torch.nn as nn
import torch.nn.functional as F

class LockedDropout(nn.Module):
    """Variational Dropout: same dropout mask across time steps"""
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or dropout == 0:
            return x
        # x: (batch_size, seq_len, emb_size)
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = mask.div_(1 - dropout)
        mask = mask.expand_as(x)
        return x * mask


class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0,
                 out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(
            emb_size, hidden_size, n_layers, batch_first=True,
            dropout=out_dropout if n_layers > 1 else 0
        )
        self.output = nn.Linear(hidden_size, output_size)

        self.emb_dropout = emb_dropout
        self.out_dropout = out_dropout

        self.locked_dropout = LockedDropout()

        if emb_size == hidden_size:
            self.output.weight = self.embedding.weight
        else:
            print("Weight tying is not possible. Hidden size should be equal to embedding size.")

    def forward(self, input_sequence):
        # Embedding + variational dropout
        emb = self.embedding(input_sequence)                          # (B, T, E)
        emb = self.locked_dropout(emb, self.emb_dropout)             # Apply variational dropout

        # LSTM
        lstm_out, _ = self.lstm(emb)

        # Output + variational dropout
        lstm_out = self.locked_dropout(lstm_out, self.out_dropout)   # Apply variational dropout
        output = self.output(lstm_out).permute(0, 2, 1)              # (B, C, T)

        return output

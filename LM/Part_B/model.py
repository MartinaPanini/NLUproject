import torch
import torch.nn as nn

class LM_LSTM_VarDROPOUT(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_VarDROPOUT, self).__init__()
        
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = nn.Dropout(emb_dropout)  # Dropout on embeddings

        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, 
                            batch_first=True, dropout=out_dropout if n_layers > 1 else 0)
        self.out_dropout = nn.Dropout(out_dropout)  # Variational Dropout on LSTM outputs

        self.projection = nn.Linear(hidden_size, emb_size, bias=False)
        self.output = nn.Linear(emb_size, output_size, bias=False)

        # Weight tying
        self.output.weight = self.embedding.weight  

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)  # (B, T, emb_size)
        emb = self.emb_dropout(emb)  # Apply dropout to embeddings (Variational)

        lstm_out, _ = self.lstm(emb)  # (B, T, hidden_size)
        lstm_out = self.out_dropout(lstm_out)  # Apply Variational Dropout on LSTM outputs

        projected = self.projection(lstm_out)  # (B, T, emb_size)
        output = self.output(projected).permute(0, 2, 1)  # (B, vocab_size, T)
        return output

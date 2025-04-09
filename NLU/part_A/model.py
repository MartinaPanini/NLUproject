import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        # Set bidirectional=True to use bidirectional LSTM
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)    
        self.slot_out = nn.Linear(hid_size * 2, out_slot)  # Multiply hid_size by 2 for bidirectional
        self.intent_out = nn.Linear(hid_size * 2, out_int)  # Multiply hid_size by 2 for bidirectional
        # Dropout layer
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance)  # utt_emb.size() = batch_size X seq_len X emb_size
        
        # pack_padded_sequence avoids computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        
        # Process the batch through the LSTM
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)
       
        # Unpack the sequence to get the actual LSTM outputs
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        
        # Get the last hidden state from both directions (concatenated)
        # The output shape for bidirectional LSTM will have the size [2 * hidden_size]
        last_hidden = torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim=1)
        
        # Compute slot logits (this includes both directions of the LSTM output)
        slots = self.slot_out(utt_encoded)
        
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0, 2, 1)  # We need this for computing the loss
        
        return slots, intent

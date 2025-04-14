import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, isDropout=False, isBidirectional=False, dropout=0.5):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size

        self.isBidirectional = isBidirectional
        self.isDropout = isDropout
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=self.isBidirectional, batch_first=True)    

        if self.isDropout:
            self.dropout = nn.Dropout(dropout) 
        if self.isBidirectional:
            hid_size=hid_size*2
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)

    def forward(self, utterance, seq_lengths):
            # utterance.size() = batch_size X seq_len
            utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size
            
            # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
            
            packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
            # Process the batch
            packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 

            # Unpack the sequence
            utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

            # Apply dropout
            if self.isDropout:
                utt_encoded = self.dropout(utt_encoded)

            # Get the last hidden state
            if self.utt_encoder.bidirectional:
                last_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
            else:
                last_hidden = last_hidden[-1,:,:]
            
            # Compute slot logits
            slots = self.slot_out(utt_encoded)
            # Compute intent logits
            intent = self.intent_out(last_hidden)
            
            # Slot size: batch_size, seq_len, classes 
            slots = slots.permute(0,2,1) # We need this for computing the loss
            # Slot size: batch_size, classes, seq_len
            return slots, intent

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight.data)
    if isinstance(m, (nn.LSTM, nn.GRU)):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim

# Funzione per l'inizializzazione dei pesi
def init_weights(mat):
    for m in mat.modules():
        if isinstance(m, (nn.GRU, nn.LSTM, nn.RNN)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)


# Modello BERT
class BERT(nn.Module):
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, dropout=0.7, device='cpu'):
        super().__init__()
        self.device = device
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        self.to(device)  # Sposta il modello sul dispositivo

    def forward(self, utterances, attentions=None, token_type_ids=None):
        outputs = self.bert_model(utterances, attention_mask=attentions, token_type_ids=token_type_ids)
        squence_output = outputs[0]
        pooled_output = outputs[1]

        # Apply dropout
        squence_output = self.dropout(squence_output)
        pooled_output = self.dropout(pooled_output)

        # Compute slots
        slots = self.slot_out(squence_output)
        # Compute intents
        intent = self.intent_out(pooled_output)

        slots = slots.permute(0, 2, 1)  # We need this for computing the loss
        
        return slots, intent


# Modello LSTM
class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, bidirectional=False, dropout_mode=False, dropout=0.7, device='cpu'):
        super(ModelIAS, self).__init__()
        
        self.device = device
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=bidirectional, batch_first=True)    
        self.dropout_mode = dropout_mode
        if self.dropout_mode:
            self.dropout = nn.Dropout(dropout)  # Apply dropout if needed
        if bidirectional:
            hid_size = hid_size * 2
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        self.to(device)  # Sposta il modello sul dispositivo

    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance)  # utt_emb.size() = batch_size X seq_len X emb_size
        
        # pack_padded_sequence to avoid computation over pad tokens
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        # Apply dropout if needed
        if self.dropout_mode:
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
        slots = slots.permute(0, 2, 1)  # We need this for computing the loss
        return slots, intent
from transformers import BertModel
import torch
import torch.nn as nn

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
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    torch.nn.init.constant_(param, 0)

class BERT(nn.Module):
    def __init__(self, hid_size, out_slot, out_intent, emb_size, vocab_len, pad_index=0):
        super(BERT, self).__init__()
        self.pad_index = pad_index
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, out_slot)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, out_intent)

    def forward(self, utterances, lengths):
        outputs = self.bert(
            input_ids=utterances,
            attention_mask=(utterances != self.pad_index).float()
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        slot_logits = self.slot_classifier(sequence_output)
        intent_logits = self.intent_classifier(pooled_output)
        return slot_logits, intent_logits

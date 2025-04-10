import torch.nn as nn
from transformers import BertModel

class ModelBert(nn.Module):

    def __init__(self, config, out_slot, out_int, ignore_list):
        super(ModelBert, self).__init__()
        
        self.num_intents = out_int
        self.num_slot = out_slot
        self.ignore_list = ignore_list
        
        self.bert = BertModel(config)
        self.slot_out = nn.Linear(config.hidden_size, out_slot)
        self.intent_out = nn.Linear(config.hidden_size, out_int)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, attention_mask, input_ids, token_type_ids):
        
        bert_out = self.bert(attention_mask=attention_mask, input_ids=input_ids, token_type_ids=token_type_ids)
        # get the last hidden states for slots and the pooled output for intents
        last_hidden_states = bert_out.last_hidden_state
        pooled_output = bert_out.pooler_output
        
        drop_slot = self.dropout(last_hidden_states)
        drop_intent = self.dropout(pooled_output)
        
        slot_out = self.slot_out(drop_slot)
        intent_out = self.intent_out(drop_intent)
        
        slot_out = slot_out.permute(0,2,1)
    
        return intent_out, slot_out
        
        
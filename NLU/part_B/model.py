import torch.nn as nn
from transformers import BertModel

class ModelBert(nn.Module):
    def __init__(self, hidden_dim, num_slot_labels, num_intent_labels, dropout_prob=0.1):
        super(ModelBert, self).__init__()

        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        
        self.dropout = nn.Dropout(dropout_prob)
        
        self.slot_classifier = nn.Linear(hidden_dim, num_slot_labels)
        self.intent_classifier = nn.Linear(hidden_dim, num_intent_labels)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        
        bert_outputs = self.bert_encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        token_embeddings = self.dropout(bert_outputs.last_hidden_state)  # shape: (batch, seq_len, hidden_dim)
        cls_embedding = self.dropout(bert_outputs.pooler_output)        # shape: (batch, hidden_dim)
        
        # Slot predictions
        slots = self.slot_classifier(token_embeddings)  # shape: (batch, seq_len, num_slot_labels)
        slots = slots.permute(0, 2, 1)  # shape: (batch, num_slot_labels, seq_len)

        # Intent prediction
        intent = self.intent_classifier(cls_embedding)  # shape: (batch, num_intent_labels)
        
        return slots, intent
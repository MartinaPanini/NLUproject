from transformers import BertModel
import torch
import torch.nn as nn

class BERT(nn.Module):
    def __init__(self, model_name, num_intents, num_slots):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intents)
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, num_slots)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden)
        pooled_output = outputs.pooler_output  # [CLS] token → intent

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        return intent_logits, slot_logits

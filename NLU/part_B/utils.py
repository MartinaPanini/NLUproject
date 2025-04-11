import torch
import torch.utils.data as data
import json
from pprint import pprint
from collections import Counter
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "mps")
PAD_TOKEN = 0
CLS_TOKEN = 101
SEP_TOKEN = 102


def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

def collate_fn(data):
    def merge(sequences):
        '''
        Merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        
        # Verifica che lengths non sia vuota prima di chiamare max()
        max_len = max(lengths) if lengths else 1
        print(f"Max sequence length: {max_len}")  # Debug

        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        padded_seqs = padded_seqs.detach()  # Remove from computational graph
        return padded_seqs, lengths
        
    new_item = {}
    try:
        # Verifica che le chiavi siano presenti nei dati
        for key in ['utterance', 'slots', 'intent']:
            if key not in data[0]:
                print(f"Warning: Missing key '{key}' in input data!")
        
        # We need only one length for packed padded sequences
        src_utt, _ = merge([item['utterance'] for item in data])
        y_slots, y_lengths = merge([item["slots"] for item in data])
        intent = torch.LongTensor([item["intent"] for item in data])

        # Ensure attention and token_type_id are always present
        attention, _ = merge([item.get("attention", []) for item in data])  # If no attention, use empty list
        token_type_id, _ = merge([item.get("token_type_id", []) for item in data])  # If no token_type_id, use empty list
        
        src_utt = src_utt.to(device)
        y_slots = y_slots.to(device)
        intent = intent.to(device)
        y_lengths = torch.LongTensor(y_lengths).to(device)
        
        new_item["utterances"] = src_utt
        new_item["intents"] = intent
        new_item["y_slots"] = y_slots
        new_item["slots_len"] = y_lengths

        # Ensure that attentions and token_type_ids are included
        new_item["attentions"] = attention
        new_item["token_type_ids"] = token_type_id
        return new_item
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        return {}

class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
      
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab


import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

def get_dev(tmp_train_raw, portion=0.10):
    portion = portion

    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev
    
    return train_raw, dev_raw


class IntentsAndSlots (data.Dataset):
    def __init__(self, dataset, lang, tokenizer, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.tokenizer = tokenizer
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

        # Generate the utterance, slot ids, attention mask, and token type ids
        self.utt_ids, self.slot_ids, self.attention_mask, self.token_id = self.mapping_seq(self.utterances, self.slots, lang.slot2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        attention = torch.Tensor(self.attention_mask[idx])  # Attention mask
        sample = {'utterance': utt, 'slots': slots, 'intent': intent, 'attention': attention}
        return sample
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, utterance, slots, slot_mapper):
        res_utterance = []
        res_slots = []
        res_attention_mask = []  # List for attention mask
        res_token_id = []

        for seq, slot in zip(utterance, slots):
            tmp_utterance = []
            tmp_slots = []
            tmp_attention_mask = []  # Add attention mask for each sequence
            tmp_token_type_id = []
            for word, element in zip(seq.split(), slot.split()):
                word_tokens = self.tokenizer(word)
                tmp_utterance.extend(word_tokens["input_ids"])
                tmp_slots.append(slot_mapper[element])
                tmp_slots.extend([slot_mapper['pad']] * (len(word_tokens["input_ids"]) - 1))

                for i in range(len(word_tokens["input_ids"])):
                    tmp_attention_mask.append(1)  # Add attention mask
                    tmp_token_type_id.append(0)

            res_utterance.append(tmp_utterance)
            res_slots.append(tmp_slots)
            res_attention_mask.append(tmp_attention_mask)  # Append attention mask
            res_token_id.append(tmp_token_type_id)

        return res_utterance, res_slots, res_attention_mask, res_token_id

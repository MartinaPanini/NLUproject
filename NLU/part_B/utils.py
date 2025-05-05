import json
from pprint import pprint
from collections import Counter
import torch
import torch.utils.data as data
import os

def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': 0}
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
            vocab['pad'] = 0
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab
    def load(self, s2id, i2id):
        self.slot2id = s2id
        self.intent2id = i2id
        self.id2slot = {v:k for k, v in s2id.items()}
        self.id2intent = {v:k for k, v in i2id.items()}


class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, tokenizer, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        # ============ CHANGES FOR BERT ================
        for x in dataset:
            self.utterances.append("[CLS] " + x['utterance'] + " [SEP]")
            self.slots.append("pad " + x['slots'] + " pad")
            self.intents.append(x['intent'])
        # ============================================

        #Create maps with utterance - slots and create also attention mask and token_type_id
        self.utt_ids, self.slots_ids, self.attention_mask, self.token_type_id = self.mapping_seq(self.utterances, self.slots, tokenizer, lang.slot2id) 
        
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slots_ids[idx])
        intent = self.intent_ids[idx]
        attention = torch.Tensor(self.attention_mask[idx])
        token_type_id = torch.Tensor(self.token_type_id[idx])
        sample = {'utterance': utt, 'slots': slots, 'intent': intent, 'attention': attention, 'token_type_id': token_type_id}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, utterance, slots, tokenizer, mapper_slot): 
        
        res_utterance = []
        res_slots = []
        res_attention = []
        res_token_type_id = []

        for sequence, slot in zip(utterance, slots):

            tmp_utt = []
            tmp_slot = []
            tmp_attention = []
            tmp_token_type_id = []

            #tokenize word per word with bert_tokenizer
            for word, element in zip(sequence.split(), slot.split(' ')):
                tmp_attention.append(1)
                tmp_token_type_id.append(0)

                #use bert tokenizer
                word_tokens = tokenizer(word)
                
                #remove CLS and SEP tokens
                word_tokens = word_tokens[1:-1]
                 
                tmp_utt.extend(word_tokens["input_ids"])

                #add true id to first word, others 'pad'
                tmp_slot.extend([mapper_slot[element]] + [mapper_slot['pad']] * (len(word_tokens["input_ids"]) - 1))

                # create attention mask 
                for i in range(len(word_tokens["input_ids"])-1):
                    tmp_attention.append(1)
                    tmp_token_type_id.append(0)

            res_utterance.append(tmp_utt)
            res_slots.append(tmp_slot)
            res_attention.append(tmp_attention)
            res_token_type_id.append(tmp_token_type_id)

        
        return res_utterance, res_slots, res_attention, res_token_type_id

def pad_sequences(sequences, pad_token='[pad]'):
    max_len = max(len(seq.split()) for seq in sequences)
    padded_sequences = [seq.split() + [pad_token] * (max_len - len(seq.split())) for seq in sequences]
    return padded_sequences

def collate_fn(data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        PAD_TOKEN = 0
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # copy each sequence into the matrix

        padded_seqs = padded_seqs.detach()  # remove these tensors from the computational graph
        return padded_seqs, lengths
    

    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    # just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    attention, _ = merge(new_item["attention"])
    token_type_id, _ = merge(new_item["token_type_id"])
    
    src_utt = src_utt.to(device) # load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    attention = attention.to(device)
    token_type_id = token_type_id.to(device)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    new_item["attentions"] = attention
    new_item["token_type_ids"] = token_type_id

    return new_item
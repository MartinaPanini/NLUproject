import json
from pprint import pprint
from collections import Counter
import torch
import torch.utils.data as data
import os
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
test_raw = load_data(os.path.join('dataset','ATIS','test.json'))

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

class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
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

        # Mappa le intent usando la mappa delle intent
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

        # Mappa le sequenze: utterance tokenizzate, slot IDs, attention mask, e token type IDs
        self.utt_ids, self.slot_ids, self.attention_mask, self.token_id = self.mapping_seq(
            self.utterances, self.slots, lang.slot2id
        )

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    #########
    # N.B → we use BERT to get word ID (token)
    #       we use LANG to obtain the ID about the slots and intents
    #
    # token_id = 0 if the token belong to the same sentence
    #            I can do this since in this case I'm not compering two sentences 
    #            Finally CLS and SEP are still present
    # subtoken → since a word can be splitted into more than one token
    #            I need to handle it. I assign the slop based on the first token
    #            and 'pad' to all the other subtokens in order to have the same length
    #########

    def mapping_seq(self, utterrance, slots, slot_mapper):
        res_utterrance = []
        res_slots = []
        res_attention_mask = []
        res_token_id = []

        for seq, slot in zip(utterrance, slots):
            tmp_utterance = []
            tmp_slots = []
            tmp_attention_mask = []
            tmp_token_type_id = []
            for word, element in zip(seq.split(), slot.split()):

                # tokenize word without special tokens
                #word_tokens = self.tokenizer(word, add_special_tokens=False)
                word_tokens = self.tokenizer(word)
                tmp_utterance.extend(word_tokens["input_ids"])
                
                #ad the id to the first token/word and the pad one for all the other tokens
                tmp_slots.append(slot_mapper[element])
                tmp_slots.extend([slot_mapper['pad']] * (len(word_tokens["input_ids"]) - 1))


                # attention mask and token type id 
                for i in range(len(word_tokens["input_ids"])):
                    tmp_attention_mask.append(1)
                    tmp_token_type_id.append(0)

        res_utterrance.append(tmp_utterance)
        res_slots.append(tmp_slots)
        res_attention_mask.append(tmp_attention_mask)
        res_token_id.append(tmp_token_type_id)

        return res_utterrance, res_slots, res_attention_mask, res_token_id
    '''
    # tokenization with Bert
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tokens = self.tokenizer.tokenize(seq)
            tmp_seq = [mapper[token] if token in mapper else mapper[self.unk] for token in tokens]
            res.append(tmp_seq)
        return res
    '''
def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        PAD_TOKEN = 0
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])

    # Build attention mask
    attention_mask, _ = merge(new_item['attetnion_mask'])
    # Build token type ids
    token_type_ids,_ = merge(new_item['token_type_ids'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_utt = src_utt.to(device)
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    attention_mask = attention_mask.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    token_type_ids = token_type_ids.to(device)

    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    new_item["attention_mask"] = attention_mask  # Add attention_mask
    new_item["token_type_ids"] = token_type_ids
    return new_item
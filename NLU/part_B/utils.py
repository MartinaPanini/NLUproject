import torch
import json
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_TOKEN = 0

def load_data(path):
    with open(path) as f:
        return json.load(f)

def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths) if lengths else 1
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lengths

    new_item = {}
    try:
        required_keys = ['utterance', 'slots', 'intent']
        if any(key not in data[0] for key in required_keys):
            raise KeyError(f"Missing keys in batch data: {required_keys}")

        src_utt, _ = merge([item['utterance'] for item in data])
        y_slots, y_lengths = merge([item['slots'] for item in data])
        intent = torch.LongTensor([item['intent'] for item in data])

        new_item["utterances"] = src_utt.to(device)
        new_item["intents"] = intent.to(device)
        new_item["y_slots"] = y_slots.to(device)
        new_item["slots_len"] = torch.LongTensor(y_lengths).to(device)
        
        return new_item
    except Exception as e:
        print(f"Collate error: {str(e)}")
        return {}

class Lang:
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2slot = {v:k for k,v in self.slot2id.items()}
        self.id2intent = {v:k for k,v in self.intent2id.items()}

    def w2id(self, elements, cutoff, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk: vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k,v in count.items():
            if v > cutoff: vocab[k] = len(vocab)
        return vocab

    def lab2id(self, elements, pad=True, unk=False):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        if unk:  # Aggiungi 'unk' se richiesto
            vocab['unk'] = len(vocab)
        for elem in elements:
            if elem not in vocab:
                vocab[elem] = len(vocab)
        return vocab

class IntentsAndSlots(Dataset):
    def __init__(self, dataset, lang, tokenizer):
        self.utterances = []
        self.slots = []
        self.intents = []
        self.lang = lang
        self.tokenizer = tokenizer
        
        for item in dataset:
            self.utterances.append(item['utterance'])
            self.slots.append(item['slots'])
            self.intents.append(item['intent'])
            
        self.utt_ids, self.slot_ids = self.mapping_seq()

    def __len__(self): return len(self.utterances)

    def __getitem__(self, idx):
        return {
            'utterance': torch.LongTensor(self.utt_ids[idx]),
            'slots': torch.LongTensor(self.slot_ids[idx]),
            'intent': self.lang.intent2id[self.intents[idx]]
        }

    def mapping_seq(self):
        utt_ids = []
        slot_ids = []
        for utt, slot in zip(self.utterances, self.slots):
            tokens = self.tokenizer(utt, add_special_tokens=False)['input_ids']
            utt_ids.append(tokens)
            slot_tags = [self.lang.slot2id.get(s, self.lang.slot2id['unk']) for s in slot.split()]
            slot_ids.append(slot_tags[:len(tokens)])
        return utt_ids, slot_ids

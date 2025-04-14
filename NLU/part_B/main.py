from functions import *
from utils import *
from model import *
from conll import evaluate
import numpy as np
import torch
import os
import json
from transformers import BertTokenizer, BertConfig
from tqdm import tqdm

if __name__ == "__main__":
    # Configuration
    bert_config = BertConfig.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading
    train_raw = load_data('dataset/ATIS/train.json')
    test_raw = load_data('dataset/ATIS/test.json')
    train_raw, dev_raw = train_test_split(train_raw, test_size=0.1, random_state=42)

    # Language processing
    words = sum([x['utterance'].split() for x in train_raw], [])
    slots = set(sum([x['slots'].split() for x in train_raw+dev_raw+test_raw], []))
    intents = set(x['intent'] for x in train_raw+dev_raw+test_raw)
    lang = Lang(words, intents, slots, cutoff=2)

    # Datasets
    train_dataset = IntentsAndSlots(train_raw, lang, tokenizer)
    dev_dataset = IntentsAndSlots(dev_raw, lang, tokenizer)
    test_dataset = IntentsAndSlots(test_raw, lang, tokenizer)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    # Model initialization
    model = BERT(
        hid_size=bert_config.hidden_size,
        out_slot=len(lang.slot2id),
        out_intent=len(lang.intent2id),
        emb_size=bert_config.hidden_size,
        vocab_len=len(lang.word2id)
    ).to(device)
    model.apply(init_weights)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    slot_criterion = nn.CrossEntropyLoss(ignore_index=lang.slot2id['pad'])
    intent_criterion = nn.CrossEntropyLoss()

    # Training loop
    best_f1 = 0
    for epoch in range(200):
        train_loss = train_loop(train_loader, optimizer, slot_criterion, intent_criterion, model)
        slot_metrics, intent_metrics, _ = eval_loop(dev_loader, slot_criterion, intent_criterion, model, lang)
        
        # Save best model
        if slot_metrics['total']['f'] > best_f1:
            best_f1 = slot_metrics['total']['f']
            torch.save(model.state_dict(), 'best_model.pt')

    # Final evaluation
    model.load_state_dict(torch.load('best_model.pt'))
    slot_metrics, intent_metrics, _ = eval_loop(test_loader, slot_criterion, intent_criterion, model, lang)
    
    print(f"Final Slot F1: {slot_metrics['total']['f']:.3f}")
    print(f"Final Intent Accuracy: {intent_metrics['accuracy']:.3f}")

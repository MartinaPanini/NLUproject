from functions import *
from utils import *
from model import *
from conll import evaluate
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import DataLoader
from torch import optim, nn
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import copy
from transformers import BertConfig, BertTokenizer  # Importa lo scheduler
import json  # Aggiunto per il salvataggio delle mappature

if __name__ == "__main__":

    bert = BertConfig.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    PAD_TOKEN = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tmp_train_raw = load_data(os.path.join('dataset/ATIS/train.json'))
    test_raw = load_data(os.path.join('dataset/ATIS/test.json'))

    portion = 0.10
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

    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev
    y_test = [x['intent'] for x in test_raw]

    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=0)

    # Crea i dataset
    train_dataset = IntentsAndSlots(train_raw, lang, tokenizer)
    dev_dataset = IntentsAndSlots(dev_raw, lang, tokenizer)
    test_dataset = IntentsAndSlots(test_raw, lang, tokenizer)

    # Crea i Dataloader
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    # ===== Training Params =====
    hid_size = 200
    emb_size = 300
    lr = 0.0001
    clip = 5
    n_epochs = 200
    runs = 5
    patience = 3

    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    ignore_list = 102
    all_slot_f1s = []
    all_intent_accs = []
    best_model = None

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)
    
    # Train loop
    for x in tqdm(range(0, runs)):
        model = BERT(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        #optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()
        
        slot_f1s, intent_acc = [], []
        for x in tqdm(range(1,n_epochs)):
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                            criterion_intents, model, clip=clip)
            if x % 5 == 0: # We check the performance every 5 epochs
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                            criterion_intents, model, lang)
                losses_dev.append(np.asarray(loss_dev).mean())
                
                f1 = results_dev['total']['f']
                # For decreasing the patience you can also use the average between slot f1 and intent accuracy
                if f1 > best_f1:
                    best_f1 = f1
                    # Here you should save the model
                    patience = 3
                else:
                    patience -= 1
                if patience <= 0: # Early stopping with patience
                    break # Not nice but it keeps the code clean

        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang) 
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])

    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    print('Slot F1', round(slot_f1s.mean(), 3), '+-', round(slot_f1s.std(), 3))
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))

    # ==== Save Results ====
    model_name = f"BERT_lr{lr}_ADAMW_F1_{round(np.mean(slot_f1s), 3)}_INTACC_{round(np.mean(intent_acc), 3)}"
    result_path = os.path.join("Results", model_name)
    os.makedirs(result_path, exist_ok=True)
    print(f"Results saved in {result_path}")

    # ===== Save Model, Tokenizer, and Label Mappings =====
    print("Salvataggio del modello, tokenizer e mappature delle etichette...")

    # Salva il modello fine-tunato
    model_path = os.path.join(result_path, "bert_model")
    best_model.save_pretrained(model_path)

    # Salva il tokenizer
    tokenizer_path = os.path.join(result_path, "bert_tokenizer")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained(tokenizer_path)

    # Salva le mappature delle etichette
    labels_path = os.path.join(result_path, "labels")
    os.makedirs(labels_path, exist_ok=True)

    with open(os.path.join(labels_path, "intent2id.json"), 'w') as f:
        json.dump(lang.intent2id, f)
    with open(os.path.join(labels_path, "slot2id.json"), 'w') as f:
        json.dump(lang.slot2id, f)

    print(f"Modello salvato in: {model_path}")
    print(f"Tokenizer salvato in: {tokenizer_path}")
    print(f"Mappature etichette salvate in: {labels_path}")

    # ===== Plot Losses ====
    plt.figure()
    plt.plot(sampled_epochs, losses_train, label='Train Loss')
    plt.plot(sampled_epochs, losses_dev, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training and Validation Loss {model_name}')
    plt.savefig(result_path + '/losses.png')
    plt.close()

    # ===== Report =====
    results_txt = os.path.join(result_path, f"results_{model_name}.txt")
    with open(results_txt, "w") as file:
        file.write(f"{model_name}\n\n")
        file.write(f'epochs used: {len(sampled_epochs)} \n')
        file.write(f'number epochs: {n_epochs} \n')
        file.write(f'lr: {lr} \n')
        file.write(f'hidden_size: {hid_size} \n')
        file.write(f'embedding_size: {emb_size} \n')
        file.write(f'Slot F1: {round(np.mean(slot_f1s), 3)} ± {round(np.std(slot_f1s), 3)} \n')
        file.write(f'Intent Accuracy: {round(np.mean(intent_acc), 3)} ± {round(np.std(intent_acc), 3)} \n')
        file.write(f"Best dev F1: {round(best_f1, 3)}\n")

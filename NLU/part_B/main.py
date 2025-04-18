from functions import *
from utils import *
from model import *
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import DataLoader
from torch import optim
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import copy
from transformers import BertTokenizer, BertModel


if __name__ == "__main__":

    bidirectional = False
    dropout = False
    PAD_TOKEN = 0
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tmp_train_raw = load_data('dataset/ATIS/train.json')
    test_raw = load_data('dataset/ATIS/test.json')

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

    # Create datasets
    train_dataset = IntentsAndSlots(train_raw, tokenizer, lang)
    dev_dataset = IntentsAndSlots(dev_raw, tokenizer, lang)
    test_dataset = IntentsAndSlots(test_raw, tokenizer, lang)

    # Create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    # ===== Training Params =====
    hid_size = 768
    emb_size = 300
    lr = 0.0001
    clip = 5
    n_epochs = 200
    patience = 5

    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    all_slot_f1s = []
    all_intent_accs = []

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    model = ModelBert(hid_size, out_slot, out_int, dropout_prob=0.1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    #optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()
    
    slot_f1s, intent_acc = [], []
    # Train loop

    for x in tqdm(range(1,n_epochs)):
        loss = train_loop(train_loader, optimizer, criterion_slots, 
                        criterion_intents, model)
        if x % 1 == 0:
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang, tokenizer)
            losses_dev.append(np.asarray(loss_dev).mean())
            f1 = results_dev['total']['f']
            print(f"f1 {f1}\n")
            print(f"best f1 {best_f1}")

            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model).to(device)
                patience = patience
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patient
                break # Not nice but it keeps the code clean

    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, lang, tokenizer)
    intent_acc.append(intent_test['accuracy'])
    slot_f1s.append(results_test['total']['f'])

    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    
    # ==== Save Results ====
   
    model_name = f"BERT_F1_{round(np.mean(slot_f1s), 3)}_INTACC_{round(np.mean(intent_acc), 3)}"
    result_path = os.path.join("Results", model_name)
    os.makedirs(result_path, exist_ok=True)

    # ==== Plot Losses ====
    plt.figure(figsize=(10, 5))
    plt.plot(sampled_epochs, losses_train, label='Training Loss', marker='o')
    plt.plot(sampled_epochs, losses_dev, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training vs Validation Loss - {model_name}' if model_name else 'Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_path, "loss_plot.png"))

    # ===== Report =====
    results_txt = os.path.join(result_path, f"results_{model_name}.txt")
    with open(results_txt, "w") as file:
        file.write(f"{model_name}\n\n")
        file.write(f'epochs used: {len(sampled_epochs)} \n')
        file.write(f'number epochs: {n_epochs} \n')
        file.write(f'lr: {lr} \n')
        file.write(f'hidden_size: {hid_size} \n')
        file.write(f'embedding_size: {emb_size} \n')
        file.write(f"Slot F1: {round(np.mean(slot_f1s), 3)} +/- {round(np.std(slot_f1s), 3)} \n")
        file.write(f"Intent Accuracy: {round(np.mean(intent_acc), 3)} +/- {round(np.std(intent_acc), 3)} \n")
        file.write(f"Best dev F1: {round(best_f1, 3)}\n")
        file.close()

    
   
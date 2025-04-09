from functions import *
from utils import *
from model import *
from conll import evaluate
from sklearn.metrics import classification_report
import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import DataLoader
from torch import optim
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    PAD_TOKEN = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Dictionaries
    w2id = {'pad': PAD_TOKEN, 'unk': 1}
    slot2id = {'pad': PAD_TOKEN}
    intent2id = {}

    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=0)

    # Create datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # Create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    # Training Setup
    hid_size = 200
    emb_size = 300
    lr = 0.0001
    clip = 5
    n_epochs = 200
    runs = 5
    patience = 5
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    # Store results from all runs
    all_slot_f1s = []
    all_intent_accs = []

    # Train loop
    for x in tqdm(range(0, runs)):
        model = ModelIAS(hid_size, out_slot, out_int, emb_size, 
                        vocab_len, pad_index=PAD_TOKEN).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()
        
        slot_f1s, intent_acc = [], []

        for x in range(1,n_epochs):
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                            criterion_intents, model)
            if x % 5 == 0:
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                            criterion_intents, model, lang)
                losses_dev.append(np.asarray(loss_dev).mean())
                f1 = results_dev['total']['f']

                if f1 > best_f1:
                    best_f1 = f1
                else:
                    patience -= 1
                if patience <= 0: # Early stopping with patient
                    break # Not nice but it keeps the code clean

        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                                criterion_intents, model, lang)
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])
    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))

    # ==== Save Results ====
    model_name = f"IAS_BI_lr{lr}_ADAM_F1_{round(np.mean(slot_f1s), 3)}_INTACC_{round(np.mean(intent_acc), 3)}"
    result_path = os.path.join("Results", model_name)
    os.makedirs(result_path, exist_ok=True)

    results_txt = os.path.join(result_path, f"results_{model_name}.txt")
    with open(results_txt, "w") as f:
        f.write(f"{model_name}\n\n")
        f.write(f"Slot F1: {round(np.mean(slot_f1s), 3)} ± {round(np.std(slot_f1s), 3)}\n")
        f.write(f"Intent Accuracy: {round(np.mean(intent_acc), 3)} ± {round(np.std(intent_acc), 3)}\n")
        f.write(f"Best dev F1: {round(best_f1, 3)}\n")
        f.write(f"Training epochs: {sampled_epochs[-1] if sampled_epochs else 'N/A'}\n")

    # ==== Plot Losses ====
    plt.figure()
    plt.plot(sampled_epochs, losses_train, label='Train Loss')
    plt.plot(sampled_epochs, losses_dev, label='Dev Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Dev Loss {model_name}')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "loss_plot.png"))
    plt.close()



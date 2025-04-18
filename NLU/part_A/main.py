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


if __name__ == "__main__":

    bidirectional = False
    dropout = False
    PAD_TOKEN = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tmp_train_raw = load_data('/dataset/ATIS/train.json')
    test_raw = load_data('/dataset/ATIS/test.json')

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
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # Create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
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
    all_slot_f1s = []
    all_intent_accs = []

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)


    # Train loop
    for x in tqdm(range(0, runs)):
        model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN, isDropout=dropout, isBidirectional=bidirectional).to(device)
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        #optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()
        
        slot_f1s, intent_acc = [], []

        for x in range(1,n_epochs):
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                            criterion_intents, model)
            if x % 5 == 0:
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
                losses_dev.append(np.asarray(loss_dev).mean())
                f1 = results_dev['total']['f']

                if f1 > best_f1:
                    best_f1 = f1
                    best_model = copy.deepcopy(model).to(device)
                    patience = 3
                else:
                    patience -= 1
                if patience <= 0: # Early stopping with patient
                    break # Not nice but it keeps the code clean
        
        best_model.to(device)

        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, lang)
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])

    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    print('Slot F1', round(slot_f1s.mean(), 3), '+-', round(slot_f1s.std(), 3))
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))
   

    # ==== Save Results ====
    if dropout and  (not bidirectional):
        model_name = f"IAS_DROP_lr{lr}_ADAM_F1_{round(np.mean(slot_f1s), 3)}_INTACC_{round(np.mean(intent_acc), 3)}"
    elif bidirectional and (not dropout):
        model_name = f"IAS_BI_lr{lr}_ADAM_F1_{round(np.mean(slot_f1s), 3)}_INTACC_{round(np.mean(intent_acc), 3)}"
    elif bidirectional and dropout:
        model_name = f"IAS_BI_DROP_lr{lr}_ADAM_F1_{round(np.mean(slot_f1s), 3)}_INTACC_{round(np.mean(intent_acc), 3)}"
    elif (not bidirectional) and (not dropout):
        model_name = f"IAS_lr{lr}_ADAM_F1_{round(np.mean(slot_f1s), 3)}_INTACC_{round(np.mean(intent_acc), 3)}"
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
    plt.grid(False)
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

    
   
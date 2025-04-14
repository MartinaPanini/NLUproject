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
import os
import json
from transformers import BertTokenizer, BertConfig
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

    train_raw, dev_raw = get_dev(tmp_train_raw)

    # Language processing
    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=0)

    train_dataset = IntentsAndSlots(train_raw, lang, tokenizer)
    dev_dataset = IntentsAndSlots(dev_raw, lang, tokenizer)
    test_dataset = IntentsAndSlots(test_raw, lang, tokenizer)

    # Dataloader instantiations
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

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

    # Train loop
    for x in tqdm(range(0, runs)):
        model = BERT(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=lr)
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
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = copy.deepcopy(model)  # Salva il modello migliore
                    patience = 3
                else:
                    patience -= 1
                if patience <= 0: # Early stopping with patience
                    break

        # Evaluation on the test set
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang) 
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])

    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    print('Slot F1', round(slot_f1s.mean(), 3), '+-', round(slot_f1s.std(), 3))
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))

    # ==== Save Results ====
    model_name = f"BERT_lr{lr}_F1_{round(np.mean(slot_f1s), 3)}_INTACC_{round(np.mean(intent_acc), 3)}"
    result_path = os.path.join("Results", model_name)
    os.makedirs(result_path, exist_ok=True)
    print(f"Results saved in {result_path}")

    # ===== Save Model, Tokenizer, and Label Mappings =====
    print("Salvataggio del modello, tokenizer e mappature delle etichette...")

    # Salva il modello
    model_path = os.path.join(result_path, "bert_model")
    torch.save(best_model.state_dict(), model_path)

    # Salva il tokenizer
    tokenizer_path = os.path.join(result_path, "bert_tokenizer")
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
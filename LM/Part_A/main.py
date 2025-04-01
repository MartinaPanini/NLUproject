from utils import *
from model import *
from functions import *
import torch
from torch.utils.data import DataLoader
from functools import partial
import torch.optim as optim
import math
import numpy as np
import copy
from tqdm import tqdm
import itertools
import os

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data
    train_raw = read_file("dataset/ptb.train.txt")
    dev_raw = read_file("dataset/ptb.valid.txt")
    test_raw = read_file("dataset/ptb.test.txt")

    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # Dataloader instantiation
    train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
####################################################################################################################################################################
    hid_sizes = [100]
    emb_sizes = [500]
    lrs = [0.0001, 0.001, 0.1]
    clips = [5]
    n_epochs_list = [100]
    patience_list = [3]
    batch_train_list = [64]
    batch_dev_test_list = [128]
####################################################################################################################################################################
    for hid_size, emb_size, lr, clip, n_epochs, patience, batch_train, batch_dev_test in itertools.product(
                hid_sizes, emb_sizes, lrs, clips, n_epochs_list, patience_list, batch_train_list, batch_dev_test_list):
        vocab_len = len(lang.word2id)
        
        #model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
        #model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
        model = LM_LSTM_DROPOUT(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
        model.apply(init_weights)

        #optimizer = optim.SGD(model.parameters(), lr=lr)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
        criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

        losses_train = []
        losses_dev = []
        sampled_epochs = []
        perplexities = []
        best_ppl = math.inf
        best_model = None
        pbar = tqdm(range(1,n_epochs+1))
        #If the PPL is too high try to change the learning rate
        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    

            if epoch % 1 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                perplexities.append(ppl_dev)
                pbar.set_description(f"PPL: {ppl_dev:.2f} | LR: {lr:.5f} | hid_size: {hid_size} | emb_size: {emb_size} | batch_train: {batch_train} | batch_dev_test: {batch_dev_test}")

                if  ppl_dev < best_ppl: 
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to(device)
                    patience = patience
                else:
                    patience -= 1
                    
                if patience <= 0: # Early stopping with patience
                    break # Not nice but it keeps the code clean

        best_model.to(device)
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
        print('Test ppl: ', final_ppl)

        model_name = f"LSTM_DROPOUT_ADAMW_PPL_{final_ppl:.2f}_LR_{lr}"  # Placeholder for final PPL
        result_path=os.path.join("Results", model_name)
        os.makedirs(result_path, exist_ok=True)

        with open(os.path.join(result_path, f"results_{model_name}.txt"), "a") as f:
            f.write(f"\n{model_name}\n\n")
            f.write(f"hid_size={hid_size}, \nemb_size={emb_size}, \nlr={lr}, \nclip={clip}, \nn_epochs={n_epochs}, \npatience={patience}, \nbatch_train={batch_train}, \nbatch_dev_test={batch_dev_test}, \ntest_ppl={final_ppl}\n")
    

        # Loss plot
        loss_dir = os.path.dirname(os.path.join(result_path, "loss_plot.png"))
        os.makedirs(loss_dir, exist_ok=True)
        loss_path = os.path.join(result_path, "loss_plot.png")
        plot_loss(sampled_epochs, losses_train, losses_dev, loss_path, model_name=model_name)
        # Perplexity plot
        ppl_dir = os.path.dirname(os.path.join(result_path, "perplexity_plot.png"))
        os.makedirs(ppl_dir, exist_ok=True)
        ppl_path = os.path.join(result_path, "perplexity_plot.png")
        plot_perplexity(sampled_epochs, perplexities, ppl_path, model_name=model_name)

        # To save the model
        path = os.path.join(result_path, f'{model_name}.pt')
        torch.save(model.state_dict(), path)
        # model = LM_LSTM_DROPOUT(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
        # model.load_state_dict(torch.load(path))
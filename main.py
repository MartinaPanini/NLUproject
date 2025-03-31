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
    # Experiment also with a smaller or bigger model by changing hid and emb sizes 
    # A large model tends to overfit
    hid_size = 200
    emb_size = 300

    # Don't forget to experiment with a lower training batch size
    # Increasing the back propagation steps can be seen as a regularization step

    # With SGD try with an higher learning rate (> 1 for instance)
    lr = 0.0001 # This is definitely not good for SGD
    clip = 5 # Clip the gradient
####################################################################################################################################################################
    vocab_len = len(lang.word2id)

    model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

####################################################################################################################################################################
    n_epochs = 10
    patience = 3
####################################################################################################################################################################
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
                
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)

# To save the model
# path = 'model_bin/model_name.pt'
# torch.save(model.state_dict(), path)
# To load the model you need to initialize it
# model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
# Then you load it
# model.load_state_dict(torch.load(path))
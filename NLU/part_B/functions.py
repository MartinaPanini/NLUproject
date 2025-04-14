import os
import torch
import torch.nn as nn
from conll import evaluate
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Function to train the model
def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []

    for sample in data:

        optimizer.zero_grad() # Zeroing the gradient
        # print(sample['utterances'])
        slots, intent = model(sample['utterances'], sample["attentions"], sample["token_type_ids"])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot

        loss_array.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return loss_array


# Function to evaluate the model
def eval_loop(data, criterion_slots, criterion_intents, model, lang, tokenizer):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    ref_slots_no_pad = [] #version of ref_slots without pad
    hyp_slots_no_pad = []

    with torch.no_grad():
        for sample in data:
            slots, intents = model(sample["utterances"], sample["attentions"], sample["token_type_ids"])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())

            #intent inference
            #get the highest probable class
            out_intent = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intent)

            # slot inference
            output_slots = torch.argmax(slots,dim=1)
            for id_seq, seq in enumerate(output_slots):

                # attention_mask = sample["attention"].tolist()[id_seq]
                length = sample['slots_len'].tolist()[id_seq]

                utt_ids = sample['utterance'][id_seq][:length].tolist()
                utterance = tokenizer.convert_ids_to_tokens(utt_ids)

                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]

                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                
                to_decode = seq[:length].tolist()  
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
        


        #remove pad tokens from reference and hypothesis for evaluation
        for refs, hyps in zip(ref_slots, hyp_slots):
            tmp_ref = []
            tmp_hyp = []
            for elem_ref, elem_hyp in zip(refs, hyps):
                if elem_ref[1] != 'pad':  #check if pad continue
                    tmp_ref.append(elem_ref)
                    tmp_hyp.append(elem_hyp)
            
            ref_slots_no_pad.append(tmp_ref)
            hyp_slots_no_pad.append(tmp_hyp)

    try:            
        results = evaluate(ref_slots_no_pad, hyp_slots_no_pad)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    
    return results, report_intent, loss_array


# Function to initialize the weights of the model
def init_weights(mat):
    for n, m in mat.named_modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            # Initialize the weights of the linear layer
            if type(m) in [nn.Linear]:
                if 'slot_out' in n or 'intent_out' in n:
                    torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                    if m.bias != None:
                        m.bias.data.fill_(0.01)



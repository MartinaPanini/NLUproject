from conll import evaluate
from sklearn.metrics import classification_report
import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    '''
    Train the model on the training set
    '''
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() 
        intent, slots = model(sample['attention_mask'], sample['utterances'], sample['token_type_ids'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot 
        loss_array.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() 
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    '''
    Evaluate the performance of the model on the validation set
    '''
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    ref_slots_pad = []
    hyp_slots_pad = []
    with torch.no_grad(): 
        for sample in data:
            intents, slots = model(sample['attention_mask'], sample['utterances'], sample['token_type_ids'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            
            # Intent inference
            out_intents = [lang.id2intent[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                utt_ids = [int(elem) for elem in utt_ids]
                gt_ids = sample['y_slots'][id_seq].tolist()
                # get the tokens of the utterance by converting the ids
                utterance = [tokenizer.convert_ids_to_tokens(elem) for elem in utt_ids]
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                to_decode = seq[:length].tolist()
                # get the slots from the ground truth ignoring the first and last token (pad for CLS and SEP)
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots[1:-1], start=1)])
                # delete internal padding inside the slots (pad for the words that create subtokens)
                ref_slots_pad.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots[1:-1], start=1) if elem != 'pad'])
                tmp_seq = []
                # get the slots from the model ignoring the first and last token (pad for CLS and SEP)
                for id_el, elem in enumerate(to_decode[1:-1], start=1):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    
    
    # remove tokens that are not in the reference (the one had to pad for the subtokens)
    for id_seq, seq in enumerate(ref_slots):
        tmp_seq = []
        for id_el, elem in enumerate(seq):
            if elem[1] != 'pad':
                tmp_seq.append(hyp_slots[id_seq][id_el])
        hyp_slots_pad.append(tmp_seq)
                              
    try:
        results = evaluate(ref_slots_pad, hyp_slots_pad)
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


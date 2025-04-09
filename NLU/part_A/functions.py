from conll import evaluate
from sklearn.metrics import classification_report
import torch

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterances'], sample['slots_len'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses. 
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    
    with torch.no_grad():  # Avoids the creation of a computational graph
        for sample in data:
            # Forward pass
            slots, intents = model(sample['utterances'], sample['slots_len'])
            
            # Compute losses
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            
            # Intent inference
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                hyp_slots.append([(utterance[id_el], lang.id2slot[elem]) for id_el, elem in enumerate(to_decode)])

    # Evaluate slot-level performance
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Handle the case where the model predicts a slot that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print("Discrepancy between predicted and reference slots:", hyp_s.difference(ref_s))
        results = {"total": {"f": 0}}  # If evaluation fails, return zero F1 score
        
    # Generate classification report for intent classification
    report_intent = classification_report(ref_intents, hyp_intents, zero_division=0, output_dict=True)
    
    return results, {
        'accuracy': report_intent['accuracy'],
        'all_true': ref_intents,
        'all_preds': hyp_intents
    }, loss_array

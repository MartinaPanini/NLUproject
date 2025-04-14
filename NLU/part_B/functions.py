from sklearn.metrics import classification_report
from conll import evaluate
import torch

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    losses = []
    for batch in data:
        optimizer.zero_grad()
        slots, intent = model(batch['utterances'], batch['slots_len'])
        
        loss_slots = criterion_slots(slots.view(-1, slots.shape[-1]), batch['y_slots'].view(-1))
        loss_intent = criterion_intents(intent, batch['intents'])
        total_loss = loss_slots + loss_intent
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        losses.append(total_loss.item())
    return losses

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    losses, ref_intents, hyp_intents = [], [], []
    ref_slots, hyp_slots = [], []
    
    with torch.no_grad():
        for batch in data:
            slots, intent = model(batch['utterances'], batch['slots_len'])
            
            # Loss calculation
            loss_slots = criterion_slots(slots.view(-1, slots.shape[-1]), batch['y_slots'].view(-1))
            loss_intent = criterion_intents(intent, batch['intents'])
            losses.append((loss_slots + loss_intent).item())

            # Intent prediction
            hyp_intents.extend(torch.argmax(intent, dim=1).tolist())
            ref_intents.extend(batch['intents'].tolist())

            # Slot prediction
            slot_preds = torch.argmax(slots, dim=2)
            for i in range(len(batch['slots_len'])):
                length = batch['slots_len'][i]
                hyp = [lang.id2slot[p] for p in slot_preds[i][:length].tolist()]
                ref = [lang.id2slot[l] for l in batch['y_slots'][i][:length].tolist()]
                hyp_slots.append(list(zip(range(length), hyp)))
                ref_slots.append(list(zip(range(length), ref)))

    try:
        slot_metrics = evaluate(ref_slots, hyp_slots)
    except Exception as e:
        print(f"Slot evaluation error: {str(e)}")
        slot_metrics = {'total': {'f': 0.0}}

    intent_accuracy = sum([1 if h == r else 0 for h,r in zip(hyp_intents, ref_intents)])/len(ref_intents)
    return slot_metrics, {'accuracy': intent_accuracy}, losses

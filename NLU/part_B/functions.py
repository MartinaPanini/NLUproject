from sklearn.metrics import classification_report
from conll import evaluate
import torch

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    
    for sample in data:
        optimizer.zero_grad()  # Reset del gradiente

        # Verifica che i dati necessari siano presenti
        if 'utterances' not in sample or 'slots_len' not in sample:
            print("Dati mancanti per l'input, salto il campione.")
            continue

        # Esegui la forward pass nel modello
        slots, intent = model(sample['utterances'], sample['slots_len'])
        
        # Calcola la loss per intenti e per slot
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        
        # Somma delle due loss
        loss = loss_intent + loss_slot
        loss_array.append(loss.item())
        
        loss.backward()  # Calcola il gradiente
        
        # Clipping del gradiente per evitare gradienti esplosivi
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()  # Aggiorna i pesi del modello

    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    
    with torch.no_grad():  # Disattiva il calcolo del grafo computazionale
        for sample in data:
            # Verifica che i dati siano correttamente formattati
            if 'utterances' not in sample or 'slots_len' not in sample:
                print("Dati mancanti per l'input, salto il campione.")
                continue

            # Esegui la forward pass nel modello
            slots, intents = model(sample['utterances'], sample['slots_len'])
            
            # Calcola la loss per intenti e per slot
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            
            # Inferenza sugli intenti
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Inferenza sugli slot
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterances'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                
                # Aggiungi gli slot di riferimento e quelli predetti
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = [(utterance[id_el], lang.id2slot[elem]) for id_el, elem in enumerate(to_decode)]
                hyp_slots.append(tmp_seq)

    try:
        # Calcola la valutazione per gli slot
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Gestisce i casi in cui la predizione contiene classi non nel riferimento
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))  # Mostra le differenze
        results = {"total": {"f": 0}}
    
    # Report per gli intenti
    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    
    return results, report_intent, loss_array

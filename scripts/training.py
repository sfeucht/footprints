'''
Functions used in train_probe.py and test_probe.py for training and data loading.
'''
import torch 
import wandb
import pandas as pd

from tqdm import tqdm 
import torch.nn as nn
from torch.utils.data import Dataset
from utils import acc, _topktoks, _topkprobs, ngram_size, is_entity_last, is_entity_ngram

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size, bias=False):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x):
        output = self.fc(x)
        return output

'''
Trains a probe for a single epoch and logs loss/accuracy.

Parameters:
    epoch: index of the current epoch
    probe: linear model to be trained
    train_loader: DataLoader of training data
    criterion: what loss function to use 
    optimizer: torch optimizer for linear model
    warmup_scheduler: scheduler for learning rate 
    accumulate: how many batches to wait until updating / clearing grads
    clip_threshold: threshold for gradient clipping. 
    batches_seen: no. batches seen before this epoch 
    device: device of Llama model  

Returns:
    None
'''
def train_epoch(epoch, probe, train_loader, criterion, optimizer, warmup_scheduler, accumulate, clip_threshold, batches_seen, device):
    probe.train()

    for batch_idx, (hidden_states, target_embs, target_toks, _, _, _) in enumerate(train_loader):
        hidden_states, target_toks = hidden_states.to(device), target_toks.to(device)
        assert(not torch.isnan(hidden_states).any() and not torch.isinf(hidden_states).any())

        if target_embs is not None:
            target_embs = target_embs.to(device)
            assert(not torch.isnan(target_embs).any() and not torch.isinf(target_embs).any())

        # get probe predictions and convert to toks if needed 
        output = probe(hidden_states.float()).to(device)
        
        output_toks = output
        
        # then calculate CELoss with the tokens 
        loss = criterion(output_toks, target_toks, reduction="mean") # removed .float()
        loss.backward() 
        
        if batch_idx % accumulate == 0 and batch_idx > 0:
            torch.nn.utils.clip_grad_norm_(probe.parameters(), clip_threshold)
            optimizer.step()
            optimizer.zero_grad()

        loss = loss.detach().item()

        # learning rate warmup for AdamW. 
        if warmup_scheduler is not None:
            if batch_idx < len(train_loader)-1:
                with warmup_scheduler.dampening():
                    pass

        # print training accuracy/loss every 10 epochs, and on the last epoch
        if batch_idx % max(accumulate, 10) == 0 or batch_idx == len(train_loader) - 1:
            train_acc = acc(output_toks.cpu(), target_toks.cpu())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTraining Acc:{:3.3f}%\tBatch Loss: {:.6f} ({} tokens)'.format(
                epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), 
                train_acc.item(), loss, hidden_states.size()[0]))

            wandb.log({"train_loss": loss, "train_acc": train_acc, 
                       "epoch": epoch, "batches_seen": 1 + batch_idx + batches_seen})

    return 1 + batch_idx + batches_seen


'''
Given a trained LinearModel and a DataLoader of test examples, test the model on a given test_loader and return csv of results (optionally)
'''
def test(probe, test_loader, criterion, device, tokenizer, target_idx, return_results=False):
    probe.eval()
    total_loss = 0.0
    total_toks = 0
    correct = 0
    topk_correct = 0
    results = []

    n_entity_ngrams = 0
    n_entity_ngrams_correct = 0
    n_other = 0
    n_other_correct = 0
    with torch.no_grad():
        # NOTE: NOW WORKS ONLY WITH NewDataset and NewCollate. doesn't have embs anymore.
        for (data, target_toks, curr_toks, doc_idxs, entity_mask) in tqdm(test_loader):
            if data is None:
                continue 
            
            output_toks = probe(data.to(device).float()).to(device)

            loss = criterion(output_toks, target_toks.to(device), reduction="mean")
            total_loss += loss.detach().item()

            for i, v in enumerate(output_toks.cpu()):         
                doc_id = doc_idxs[i]
                current_tok = _topktoks(curr_toks[i])
                actual_tok = target_toks[i]
                predicted_tok = _topktoks(v)
                this_loss = criterion(v, target_toks[i])
                is_correct = predicted_tok == actual_tok

                total_toks += 1
                if is_correct:
                    correct += 1
                if actual_tok in _topktoks(v, k=5): # less interpretable NOW THAT PINV
                    topk_correct += 1
                    
                kl_divergence = -1
                q_target_log_prob = torch.inf
                p_target_log_prob = torch.inf
                

                if entity_mask is not None: 
                    this_is_entity_ngram = bool(is_entity_ngram(i, entity_mask, target_idx=target_idx))
                    if this_is_entity_ngram:
                        # print(tokenizer.decode(current_tok.tolist()), tokenizer.decode(actual_tok.tolist()))
                        n_entity_ngrams += 1
                        n_entity_ngrams_correct += int(is_correct)
                    else: # count up coarse "other" values. 
                        n_other += 1
                        n_other_correct += int(is_correct)
                    
                    n = 0 
                    this_is_entity_last = is_entity_last(i, entity_mask)
                    if this_is_entity_last: # if entity save how big the entity is 
                        n = ngram_size(i, entity_mask)
                    
                if return_results:
                    # BOS token becomes encoded as NaN in pandas here
                    curr_result = {
                        "doc_id" : doc_id.item(),
                        "current_tok_id" : current_tok.item(),
                        "actual_tok_id" : actual_tok.item(),
                        "predicted_tok_id" : predicted_tok.item(),
                        "current_tok" : tokenizer.decode(current_tok.tolist()),
                        "actual_tok" : tokenizer.decode(actual_tok.tolist()),
                        "predicted_tok" : tokenizer.decode(predicted_tok.tolist()),
                        "loss" : this_loss.item(),

                        # from this information you can split into entity_last, entity_notlast, and nonentity. 
                        # as well as the case "entity_ngram": whether this probe is predicting the first tok FROM the last tok of an entity. 
                        "is_entity" : entity_mask[i].item() if entity_mask is not None else -1,
                        "is_entity_last" : this_is_entity_last if entity_mask is not None else -1,
                        "is_entity_ngram" : this_is_entity_ngram if entity_mask is not None else -1,
                        "n" : n,

                        "kl_divergence" : kl_divergence,
                        "q_target_log_prob" : q_target_log_prob,
                        "p_target_log_prob" : p_target_log_prob,
                        **_topkprobs(v, tokenizer)
                    }

                    results.append(curr_result)
            
                 
    test_loss = total_loss / len(test_loader) # divide total average loss by no. batches
    test_acc = 100 * correct / total_toks
    topk_acc = 100 * topk_correct / total_toks

    if n_entity_ngrams > 0:
        entity_ngram_acc = 100 * n_entity_ngrams_correct / n_entity_ngrams
    else:
        entity_ngram_acc = -1
    
    if n_other > 0:
        other_acc = 100 * n_other_correct / n_other
    else:
        other_acc = -1
    
    if return_results:
        return test_loss, test_acc, topk_acc, entity_ngram_acc, other_acc, pd.DataFrame(results)
    else:
        return test_loss, test_acc, topk_acc, entity_ngram_acc, other_acc

'''
Dataset for retrieving tokenized documents from csv along with masks that mark
which tokens correspond to "entities" 
'''
class DocDataset(Dataset):
    def __init__(self, model, tokenizer, layer_name, target_idx, dataset_csv, window_size, vocab_size, device, entities=None):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_name = layer_name # int: -1 is embedding, 0-27 for layers, 28 for logits right at the end
        self.target_idx = target_idx # -1 is previous token, 0 is current. 
        self.dataset_csv = dataset_csv
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.device = device
        self.entities = entities # list of strings that are the entities we want to mask out

        if self.entities is not None:
            self.entities = [self.tokenize(e, bos=False) for e in self.entities]
    
    # llama tokenizer already adds BOS token, clip to window size 
    def tokenize(self, text, bos=True):
        if bos:
            t = self.tokenizer(text)['input_ids']
        else:
            t = self.tokenizer(text)['input_ids'][1:]
        
        if len(t) > self.window_size:
            return t[:self.window_size]
        else:
            return t
    
    # iterate through sequence and mark subseq occurs in sequence 
    def mask_iterator(self, sequence, subseq, mask):
        sequence = list(sequence.cpu())
        if len(subseq) <= len(sequence):
            for i in range(len(sequence)-len(subseq)+1):
                assert len(sequence[i:i+len(subseq)]) == len(subseq)
                if (sequence[i:i+len(subseq)] == subseq):
                    mask[i:i+len(subseq)] = 1
        return torch.Tensor(mask)
    
    # returns number of documents, not tokens 
    def __len__(self):
        if self.dataset_csv is not None: 
            return len(self.dataset_csv) 

    # get document tokens and mask 
    def __getitem__(self, index):
        doc = self.dataset_csv.iloc[index]
        doc_string = str(doc['text'])
        tokens = torch.tensor(self.tokenize(doc_string))

        entity_mask = torch.zeros_like(tokens)
        if self.entities is not None:
            for e in self.entities:
                entity_mask = self.mask_iterator(tokens, e, entity_mask)
        
        return torch.tensor(index), doc_string, tokens, entity_mask

'''
Bloated collate function that takes sequences and retrieves hidden states for the given model using nnsight
'''
class DocCollate(object):
    def __init__(self, layer, target_idx, tokenizer, model, window_size, device):
        self.layer = layer
        self.target_idx = target_idx
        self.tokenizer = tokenizer
        self.model = model
        self.window_size = window_size
        self.device = device
        
    def __call__(self, batch):
        # first, get all the hidden states by doing a giant Trace
        strings = [s for (_, s, _, _) in batch]

        with self.model.trace(strings):
            if self.layer == -1:
                states = self.model.model.embed_tokens.output.save()
            elif self.layer == 32:
                states = self.model.model.norm.output.save()
            else:
                states = self.model.model.layers[self.layer].output[0].save()

        # get the mask for padding that was added by nnsight 
        attention_mask = self.tokenizer(strings, return_tensors='pt', padding='longest')['attention_mask']

        # then loop through the entire thing to keep same logic for embs, tokens and doc_idxs 
        source_hss = []
        target_toks = []
        current_toks = []
        doc_idxs = []
        entity_masks = []
        for i, doc in enumerate(batch): 
            # batch looks like [doc0:(0, text, tokens, mask), doc1:(1, text, tokens, mask)...]
            doc_idx, tokens, entity_mask = (a.cpu() for a in doc if type(a)!=str)

            # get the hidden states we just calculated, and trim off the PAD tokens
            # for llama 2 7b padding is always at the beginning 
            hidden_states = states[i][-sum(attention_mask[i]):] 
            assert (len(hidden_states) == len(tokens))

            # make sure that hidden_states has enough tokens to deal with the given target_idx.
            # if the target_idx is gonna be outside the bounds of hidden_states, we want to skip doc.
            if abs(self.target_idx) >= len(hidden_states):
                continue

            # target_idx == -1:
            #   source_hss: BOS [this is an example sentence]
            #   target_toks: [BOS this is an example] sentence
            # target_idx == -2: 
            #   (BOS this [is an) example sentence]
            # target_idx == -3:
            #   (BOS this is) [an example sentence]
            if self.target_idx < 0: 
                pos = abs(self.target_idx)
                source_hss.append(hidden_states[pos:])
                target_toks.append(tokens[:-pos])
                current_toks.append(tokens[pos:])
                doc_idxs.append(torch.tensor([doc_idx for _ in range(len(hidden_states[pos:]))], device='cpu'))
                entity_masks.append(entity_mask[pos:])
            
            # target_idx == 1:
            #   source_hss: [BOS this is an example] sentence
            #   target_toks: BOS [this is an example sentence]
            # target_idx == 2:
            #   [BOS this (is an] example sentence)
            # target_idx == 3:
            #   [BOS this is] (an example sentence)
            elif self.target_idx > 0:
                pos = abs(self.target_idx)
                source_hss.append(hidden_states[:-pos])
                target_toks.append(tokens[pos:])
                current_toks.append(tokens[:-pos])
                doc_idxs.append(torch.tensor([doc_idx for _ in range(len(hidden_states[:-pos]))], device='cpu'))
                entity_masks.append(entity_mask[:-pos])

            # exclude predicting bos_embedding -> BOS
            elif self.target_idx == 0:
                source_hss.append(hidden_states[1:])
                target_toks.append(tokens[1:])
                current_toks.append(tokens[1:])
                doc_idxs.append(torch.tensor([doc_idx for _ in range(len(hidden_states[1:]))], device='cpu'))
                entity_masks.append(entity_mask[1:])
        
        # sometimes docs are too small 
        if len(source_hss) > 0:
            source_hss = torch.cat(source_hss)
            target_toks = torch.cat(target_toks)
            current_toks = torch.cat(current_toks)
            doc_idxs = torch.cat(doc_idxs)
            entity_masks = torch.cat(entity_masks)
            return (source_hss, target_toks, current_toks, doc_idxs, entity_masks)
        else:
            return None, None, None, None, None
'''
Small helper functions used for training and testing.
'''
import torch

def acc(pred_dist, target_toks):
    return 100 * (sum(pred_dist.argmax(dim=-1) == target_toks) / len(target_toks))

def _topktoks(logits, k=1):
    _, top_tokens = logits.topk(k=k, dim=-1)
    return top_tokens 

def _topkprobs(logits, tokenizer, k=5):
    top_probs, top_tokens = torch.softmax(logits, dim=0).topk(k=k, dim=-1)
    out = {}
    for i in range(k):
        out[f"top_{i+1}_prob"] = top_probs[i].item()
        out[f"top_{i+1}_tok_id"] = top_tokens[i].item()
        out[f"top_{i+1}_tok"] = tokenizer.decode(top_tokens[i].tolist())
    return out

'''
The next three functions handle entity_mask lists for the function test() in training.py 
'''
# assume you are given i as the last position in ngram mask
def ngram_size(i, entity_mask):
    assert i >= 0
    assert entity_mask[i] == 1 
    assert i + 1 == len(entity_mask) or entity_mask[i+1] == 0

    if i == 0:
        return 1 # unigram [1]
    size = 1
    j = i - 1 

    while entity_mask[j] == 1:
        size += 1
        j -= 1 
        if j < 0:
            break 
    return size 

# is position i the last position in an entity in this mask? 
def is_entity_last(i, entity_mask):
    if entity_mask[i] == 0:
        return False
    else:
        if i + 1 == len(entity_mask):
            return True # end of sequence, i is last. 

        # otherwise i is last if the next token is 0. 
        return entity_mask[i+1] == 0

# this is about to get thorny. 
# returns whether there is an entity ngram of our chosen shape at i based on target index:
#   0 counts only unigram entities, e.g. [Iran]
#   -1 counts only bigram entities, e.g. [New, York]
#   -2 counts only trigram entities, e.g. [Empire, State, Building]
#   -3 counts only 4-gram entities, e.g. [Co, ca, Co, la]
#   1 counts only bigram entities but the other way. 
def is_entity_ngram(i, entity_mask, target_idx=-1):
    if entity_mask[i] == 0:
        return False
    
    # otherwise, the current guy is an entity, so let's check if the previous/future one
    # (to make an ngram) and the ones in between are also an entity as well 
    else:
        if target_idx == 0:
            if i-1<0 and i+1 >= len(entity_mask): # [1]
                return bool(entity_mask[i])
            elif i-1<0 and i+1 < len(entity_mask): # [1,0...]
                return bool(entity_mask[i]) and not entity_mask[i+1]
            elif i-1>=0 and i+1 >= len(entity_mask): # [...0,1]
                return bool(entity_mask[i]) and not entity_mask[i-1]
            else: # i-1>=0 and i+1 < len(entity_mask), they're both within bounds
                return bool(entity_mask[i]) and not entity_mask[i+1] and not entity_mask[i-1]
        
        # backwards bigram. [1,1,0,0,0]
        elif target_idx == -1: 
            if i-1<0:
                return False 
            if i-2<0:
                if i+1 < len(entity_mask):
                    return not entity_mask[i+1] and bool(entity_mask[i] and entity_mask[i-1])
                else:
                    return bool(entity_mask[i] and entity_mask[i-1])
            else:
                if i+1 < len(entity_mask):
                    return not entity_mask[i+1] and not entity_mask[i-2] and bool(entity_mask[i] and entity_mask[i-1])
                else:
                    return not entity_mask[i-2] and bool(entity_mask[i] and entity_mask[i-1])
        
        # backwards trigram [0,1,1,1,0]
        elif target_idx == -2:
            if i-2<0:
                return False 
            if i-3<0:
                if i+1 < len(entity_mask):
                    return not entity_mask[i+1] and sum(entity_mask[i-2:i+1]) == 3
                else:
                    return sum(entity_mask[i-2:i+1]) == 3
            else:
                if i+1 < len(entity_mask):
                    return not entity_mask[i+1] and not entity_mask[i-3] and (sum(entity_mask[i-2:i+1]) == 3)
                else:
                    return not entity_mask[i-3] and (sum(entity_mask[i-2:i+1]) == 3)
        
        # backwards 4-gram [0,1,1,1,1]
        elif target_idx == -3:
            if i-3<0:
                return False 
            if i-4<0: 
                if i+1 < len(entity_mask): # [1,1,1,1,0...]
                    return not entity_mask[i+1] and sum(entity_mask[i-3:i+1]) == 4
                else: # [1,1,1,1]
                    return sum(entity_mask[i-3:i+1]) == 4
            else:
                if i+1 < len(entity_mask): # [0,1,1,1,1,0...]
                    return not entity_mask[i+1] and not entity_mask[i-4] and (sum(entity_mask[i-3:i+1]) == 4)
                else: # [...0,1,1,1,1]
                    return not entity_mask[i-4] and (sum(entity_mask[i-3:i+1]) == 4)
        
        # forwards bigram [0,1,1,0]
        elif target_idx == 1:
            if i+1 >= len(entity_mask): # [...,1]
                return False 
            if i+2 >= len(entity_mask): # [...,1,0]
                if i-1 < 0: # [1,0]
                    return bool(entity_mask[i] and entity_mask[i+1])
                else: # [...0,1,0]
                    return not entity_mask[i-1] and bool(entity_mask[i] and entity_mask[i+1])
            else:
                if i-1 < 0: # [1,0,...]
                    return not entity_mask[i+2] and bool(entity_mask[i] and entity_mask[i+1])
                else: # [...0,1,0,...]
                    return not entity_mask[i-1] and not entity_mask[i+2] and bool(entity_mask[i] and entity_mask[i+1])
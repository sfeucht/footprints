'''
TODO clean up this file 
Loads a probe checkpoint and gets test results on it, including training ngram frequency information.
'''
import os
import csv
import torch
import argparse 
import pandas as pd
import numpy as np
import regex as re
import spacy 

import torch.nn.functional as F
from collections import Counter
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from nnsight import LanguageModel
from modules.training import LinearModel, test, weighted_mse
from modules.state_data import NewDataset, NewCollate

import lovely_tensors as lt 
# lt.monkey_patch()

torch.manual_seed(0)

idx_to_n = {
    1 : 2,
    0 : 1,
    -1 : 2,
    -2 : 3,
    -3 : 4
}

def idx_to_zip(idx, toks):
    if idx <= 0:
        skip = {
            0 : 0,
            -1 : 1,
            -2 : 2,
            -3 : 3
        }[idx]
        return zip(toks[:-skip], toks[skip:])
    elif idx == 1:
        return zip(toks[1:], toks[:-1])

def datasetname(input):
    return input.split('/')[-1][:-4]

# take in results dataframe and add column `train_freq` indicating frequency of that ngram in the training dataset. 
# note: this only does the SKIP ngrams. so "the big tower" and "the small tower" are the same for tgtidx=-2.
def train_ngram_info(results, train_df, tokenizer, model_name, target_idx):
    # get information about train ngram frequencies
    gt_ctr = Counter()
    all_gt_ct = 0
    for d in list(train_df['text']):
        if "llama" in model_name:
            toks = tokenizer(d)['input_ids']
        else:
            bos = tokenizer.bos_token
            toks = tokenizer(bos+d)['input_ids']
        gt_ctr += Counter(idx_to_zip(target_idx, toks))
        all_gt_ct += len(toks)
    
    # then for each ngram in results, save that ngram's train count
    results['all_train_ct'] = [all_gt_ct for _ in range(len(results))]
    new = []
    for i, row in results.iterrows():
        # want to be able to change this based on target_idx
        try:
            ct = gt_ctr[(row['actual_tok_id'], row['current_tok_id'])]
        except KeyError:
            ct = 0
        
        row['ngram_train_ct'] = ct
        
        if row['ngram_train_ct'] > 0:
            row['log_ngram_train_freq'] = np.log(row['ngram_train_ct'] / row['all_train_ct'])
        else:
            row['log_ngram_train_freq'] = 0
        new.append(row)
    
    return pd.DataFrame(new)

# returns a |V|x|V| matrix where each row is conditional probability of all preceding tokens
# given that token as the succeeding token. e.g. row for "York" has a high probability on "New"
# https://jofrhwld.github.io/teaching/courses/2022_lin517/python_sessions/04_session5.html#conditional-probability
def train_conditional_probs(train_df, train_name, tokenizer, model_name, vocab_size, target_idx=-1):
    q_name = train_name + f"_qmodel{target_idx}.ckpt"
    if q_name in os.listdir("../data/qmodels"):
        # load the previous matrix for this dataset and target_idx=-1
        return torch.load(f"../data/qmodels/{q_name}")

    # edit this to account for diff target_idxs
    ng_ctr = Counter()
    ug_ctr = Counter()
    out = torch.zeros(size=(vocab_size, vocab_size))
    for doc in list(train_df['text']):
        if "llama" in model_name:
            toks = tokenizer(doc)['input_ids']
        else:
            bos = tokenizer.bos_token
            toks = tokenizer(bos+doc)['input_ids']
        ng_ctr += Counter(idx_to_zip(target_idx, toks)) # ngrams(toks, idx_to_n[target_idx]))
        ug_ctr += Counter(toks)
    assert(len(ug_ctr) <= vocab_size)
    
    # each ROW correspnds to the second token
    # right now we fill with joint probabilities 
    for tup, ct in ng_ctr.items():
        out[tup] = ct / sum(ng_ctr.values())
    
    # then divide each row by p(x1) out of all unigrams 
    unigram_probs = torch.zeros(size=(vocab_size,))
    for i in range(vocab_size):
        unigram_probs[i] = ug_ctr[i] / sum(ug_ctr.values())
    assert(torch.sum(unigram_probs) == 1)
    
    # avoid divide by 0 errors by just dividing by 1 (since numerator will also be 0 in those cases)
    denom = torch.where(unigram_probs == 0, torch.ones_like(unigram_probs), unigram_probs)
    out = (out.T / denom).T
    print(out)

    torch.save(out, f"../data/{q_name}.ckpt")
    return out
        

def main(args):
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    MODEL_NAME = args.model
    VOCAB_SIZE = 32000
    MODEL_SIZE = 4096
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LanguageModel(MODEL_NAME, device_map='cuda')
    
    if MODEL_NAME == "meta-llama/Llama-2-7b-hf":
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # from https://github.com/meta-llama/llama/blob/main/llama/model.py
    # do not trust tokenizer.model_max_length
    WINDOW_SIZE = 512 # 2048 actually
    for p in model.parameters():
        p.requires_grad = False
    
    if args.predict_embs: 
        probe = LinearModel(MODEL_SIZE, MODEL_SIZE).to(device)
    else:
        probe = LinearModel(MODEL_SIZE, VOCAB_SIZE).to(device)
    probe.load_state_dict(torch.load(args.checkpoint))

    # intuit target_idx and layer from the filename 
    target_idx = args.target_idx 
    if target_idx is None:
        if "TGTIDX" in args.checkpoint:
            s = re.search(r'TGTIDX-\d+|TGTIDX\d+', args.checkpoint).group()
            target_idx = int(s[6:])
        else:
            raise Exception("Can't infer target index from checkpoint: " + args.checkpoint)
    
    layer = args.layer 
    if layer is None:
        if "LAYER" in args.checkpoint:
            s = re.search(r'LAYER-\d+|LAYER\d+', args.checkpoint).group()
            layer = int(s[5:])
        else:
            raise Exception("Can't infer layer from checkpoint: " + args.checkpoint)
    
    random_data = args.test_data == "RANDOM"
    collate_fn = NewCollate(layer, target_idx, tokenizer, model, WINDOW_SIZE, args.residuals, random=random_data)
    if random_data:
        test_data = None
    else: 
        test_data = pd.read_csv(args.test_data)
    
    if args.test_data == "../data/expansion1_text.csv":
        test_data = test_data.loc[test_data['llama-2-7b_correct']]
        print(f"pruned down to {len(test_data)}")
    
    # pass in subjects from counterfact dataset as entities 
    which_entity = "subject"
    entities = None
    if test_data is not None:
        if which_entity in test_data.columns:
            entities = list(test_data[which_entity])
    
    if 'wikipedia' in args.test_data:
        if args.spacy:
            nlp = spacy.load("en_core_web_sm")
            entities = []
            for d in test_data['text']:
                doc = nlp(d)
                # https://stackoverflow.com/questions/70185150/return-all-possible-entity-types-from-spacy-model
                # ['ORG', 'CARDINAL', 'DATE', 'GPE', 'PERSON', 'MONEY', 'PRODUCT', 'TIME', 'PERCENT', 'WORK_OF_ART', 'QUANTITY', 'NORP', 'LOC', 'EVENT', 'ORDINAL', 'FAC', 'LAW', 'LANGUAGE']
                # we want non-number ones. no dates, money, cardinals, time etc.
                desired_types = ['ORG', 'GPE', 'PERSON', 'PRODUCT', 'WORK_OF_ART', 'NORP', 'LOC', 'EVENT', 'FAC', 'LAW', 'LANGUAGE']
                entities += [e.text for e in doc.ents if e.label_ in desired_types]
            entities = list(set(entities))
            print(entities[:10])
        else: 
            multi_tok = []
            for txt in test_data['text']:
                txt = re.sub(r'[^\w\s]', '', txt)
                txt = re.sub(r'[0-9]', '', txt)
                for word in Counter(txt.split()):
                    if len(tokenizer(word)['input_ids'][1:]) > 1:
                        multi_tok.append(word)
            print(multi_tok[:10])
            entities = list(set(multi_tok))
    
    test_dataset = NewDataset(model, tokenizer, layer, target_idx, test_data, WINDOW_SIZE, VOCAB_SIZE, device, entities=entities, rand_size=500) # 500 * 10 toks per sequence, 5,000 val
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, collate_fn=collate_fn, drop_last=True, pin_memory=False)
    
    k = 5
    criterion = {
        "weighted_mse" : weighted_mse,
        "ce" : F.cross_entropy,
        "mse" : F.mse_loss
    }[args.criterion]
    
    test_loss, test_acc, test_topk_acc, test_entity_ng_acc, test_other_acc, test_results = test(probe, test_loader, 
        criterion, device, model, tokenizer, None, args.predict_embs, target_idx, q_model=None, return_results=True, k=k)
    
    def wikiextra(test_data, spacy):
        if "wikipedia" in test_data:
            if spacy:
                return "_mte"
            else:
                return "_mtw"
        else:
            return ""

    model_folder = args.checkpoint.split('/')[2] # hf-llama-2
    run_name = args.checkpoint.split('/')[-2]
    log_dir = f"../logs/{model_folder}/{run_name}/"
    out_csv = f"{datasetname(args.test_data)}_results{wikiextra(args.test_data, args.spacy)}.csv" # assume we're testing on the same layer
    if args.model == 'epfl-llm/meditron-7b':
        out_csv = "meditron_" + out_csv
        
    os.makedirs(log_dir, exist_ok=True)
    print(log_dir + out_csv)
    test_results.to_csv(log_dir + out_csv, quoting=csv.QUOTE_ALL, encoding='utf-8')
    print('Test Loss: {:10.4f}  Accuracy: {:3.4f}%\n'.format(test_loss, test_acc))
    print('Test Top-{} Accuracy: {:3.4f}%\n'.format(k, test_topk_acc))
    print('Test Accuracy for Entity Ngrams: {:3.4f}% (Other: {:3.4f})\n'.format(test_entity_ng_acc, test_other_acc))

    return test_loss, test_acc, test_entity_ng_acc, test_other_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # information about what you want to test on
    parser.add_argument('--train_data', type=str, default='../data/train_tiny_1000.csv')
    # parser.add_argument('--test_data', type=str, default='../data/test_tiny_500.csv')
    # parser.add_argument('--test_data', type=str, default='../data/expansion1_text.csv')
    # parser.add_argument('--test_data', type=str, default='RANDOM')
    parser.add_argument('--test_data', type=str, default="../data/wikipedia_test_500.csv")
    parser.add_argument('--layer', type=int, default=None, required=False,
                        help='which layer you want to TEST on. GPT-J: from -1..28 where -1 is embedding layer and 28 is output. Llama: from -1...32 where -1 is embedding layer and 32 is output.')
    parser.add_argument('--target_idx', type=int, default=None, required=False, # help msg uses NEW numbering
                        help='which token you want to TEST predicting (e.g. 0 for current token, -1 for prev)')
    parser.add_argument('--criterion', type=str, choices=['weighted_mse', 'mse', 'ce'], default='ce')

    # information we need to load probe properly 
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--model', type=str, choices=['meta-llama/Llama-2-7b-hf', 'epfl-llm/meditron-7b'], required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--predict_embs', action='store_true')
    parser.add_argument('--residuals', action='store_true')
    parser.add_argument('--spacy', action='store_true')
    parser.set_defaults(predict_embs=False, residuals=False, spacy=False)
    
    args = parser.parse_args()
    main(args)
    
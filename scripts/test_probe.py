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
from nnsight import LanguageModel
from training import LinearModel, test, DocDataset, DocCollate

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

# get nice string when saving wikipedia results
def wikiextra(datasetname, spacy):
    if "wikipedia" in datasetname:
        if spacy:
            return "_mte"
        else:
            return "_mtw"
    else:
        return ""

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
    
    model = LanguageModel(args.model, device_map=device)
    tokenizer = model.tokenizer 
    tokenizer.add_special_tokens({'pad_token':'<s>'})

    VOCAB_SIZE = model.vocab_size
    MODEL_SIZE = model.config.hidden_size
    MODEL_NAME = args.model.split('/')[-1]
    
    # window size is actually 2048 but I choose 512 for brevity
    WINDOW_SIZE = 512 
    
    for p in model.parameters():
        p.requires_grad = False
    
    probe = LinearModel(MODEL_SIZE, VOCAB_SIZE).to(device)
    probe.load_state_dict(torch.load(args.checkpoint))

    # intuit target_idx from the filename 
    if "TGTIDX" in args.checkpoint:
        s = re.search(r'TGTIDX-\d+|TGTIDX\d+', args.checkpoint).group()
        target_idx = int(s[6:])
    else:
        raise Exception("Can't infer target index from checkpoint: " + args.checkpoint)
    
    # intuit layer from the filename 
    if "LAYER" in args.checkpoint:
        s = re.search(r'LAYER-\d+|LAYER\d+', args.checkpoint).group()
        layer = int(s[5:])
    else:
        raise Exception("Can't infer layer from checkpoint: " + args.checkpoint)
    
    collate_fn = DocCollate(layer, target_idx, tokenizer, model, WINDOW_SIZE, device)

    test_data = pd.read_csv(args.test_data)
    
    if args.test_data == "../data/counterfact_expandeds.csv":
        corr_str = {
            'Llama-2-7b-hf' : 'llama-2-7b',
            'Meta-Llama-3-8B' : 'llama-3-8b'
        }[MODEL_NAME]
        
        test_data = test_data.loc[test_data[f'{corr_str}_correct']]
        print(f"pruned down to only correct CounterFact answers, {len(test_data)}")
    
    # pass in subjects from counterfact dataset as entities 
    which_entity = "subject"
    entities = None
    if test_data is not None:
        if which_entity in test_data.columns:
            entities = list(test_data[which_entity])
    
    if 'wikipedia' in args.test_data:
        if args.wiki_spacy:
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
    
    test_dataset = DocDataset(model, tokenizer, layer, target_idx, test_data, WINDOW_SIZE, VOCAB_SIZE, device, entities=entities)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, collate_fn=collate_fn)
    
    criterion = {
        "ce" : F.cross_entropy,
        "mse" : F.mse_loss
    }[args.criterion]
    
    test_loss, test_acc, test_topk_acc, test_entity_ng_acc, test_other_acc, test_results = test(probe, test_loader, 
        criterion, device, tokenizer, target_idx, return_results=True)

    model_folder = args.checkpoint.split('/')[2] # hf-llama-2
    run_name = args.checkpoint.split('/')[-2]
    log_dir = f"../logs/{model_folder}/{run_name}/"
    out_csv = f"{datasetname(args.test_data)}_results{wikiextra(args.test_data, args.wiki_spacy)}.csv" 
        
    os.makedirs(log_dir, exist_ok=True)
    print(log_dir + out_csv)
    test_results.to_csv(log_dir + out_csv, quoting=csv.QUOTE_ALL, encoding='utf-8')
    print('Test Loss: {:10.4f}  Accuracy: {:3.4f}%\n'.format(test_loss, test_acc))
    print('Test Top-5 Accuracy: {:3.4f}%\n'.format(test_topk_acc))
    print('Test Accuracy for Entity Ngrams: {:3.4f}% (Other: {:3.4f})\n'.format(test_entity_ng_acc, test_other_acc))

    return test_loss, test_acc, test_entity_ng_acc, test_other_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # defaults 
    parser.add_argument('--criterion', type=str, choices=['mse', 'ce'], default='ce')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=12)

    # what dataset to test on 
    parser.add_argument('--test_data', type=str, 
                        choices=[
                            '../data/counterfact_expanded.csv', 
                            '../data/test_tiny_500.csv', 
                            '../data/wikipedia_test_500.csv'
                        ], default="../data/test_tiny_500.csv")
    
    # for wikipedia dataset, do MTE if True, otherwise do MTW
    parser.add_argument('--wiki_spacy', action='store_true')
    parser.set_defaults(wiki_spacy=False)
    
    # specify probe checkpoint and model. tests the same layer and target_idx. 
    parser.add_argument('--model', type=str, choices=['meta-llama/Llama-2-7b-hf', 'meta-llama/Meta-Llama-3-8B'], default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--checkpoint', type=str, required=True, help="e.g. ../checkpoints/Llama-2-7b-hf/llamaLAYER12-TGTIDX-2.../final.ckpt") 
    
    args = parser.parse_args()
    main(args)
    
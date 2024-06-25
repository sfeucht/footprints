'''
TODO brush up and comment
'''
import os
import re 
import csv
import time
import torch 
import pickle
import argparse 
import numpy as np
import pandas as pd 

from collections import Counter
from transformers import AutoTokenizer
from modules.training import LinearModel
from nnsight import LanguageModel

tgt_idxs = ['i0', 'in1', 'in2', 'in3']

def datasetname(s):
    if s == "wikipedia":
        return s 
    elif s == "bigbio/med_qa":
        return s.split('/')[-1]
    elif s == "bigbio/blurb":
        return "blurb"
    elif s == "wmt/wmt14":
        return "wmt14"
    else:
        return s.split('/')[-1][:-4]

# load in the outputs of `algorithm1.py` to double check and use as basis for this.
def load_scores(model, dataset):
    dir = f"../logs/{model}/candidates/{dataset}_layer1_9/"
    dfs = []
    for f in os.listdir(dir):
        if "scores" in f: 
            df = pd.read_csv(dir + f)
            df['doc_idx'] = int(re.search(r'\d+', f).group(0))
            dfs.append(df)
    
    return pd.concat(dfs).drop(columns=["Unnamed: 0"])


# ## The Recipe
# Okay boys, I think I found the recipe. I think that including -3 information is weird and janky but 
# including -1 and -2 information for all tokens as well as i=0 information for the last token really clinches this. 
def psi(doc_unis, i, j):
    sn = doc_unis.iloc[i:j+1]
    
    idx_to_key = {
        0 : 'tok-1_i0_probdelta',
        -1 : 'tok-1_in1_probdelta',
        -2 : 'tok-1_in2_probdelta',
        -3 : 'tok-1_in3_probdelta'
    }
    
    # we're doing 0 indexing for t, so this is different from paper 
    # in the paper we did start-end, so we have to flip all these to end-start, add - in front of probdelta
    # also, we have to do (t + idx < i) since we're using absolute idxs
    
    # OPTION removing i0 information 
    # ideal = 0.0001
    # score = 0
    ideal = 1
    score = -sn.iloc[-1]['tok-1_i0_probdelta'] 
    for t, row in sn.iterrows():
        # for idx in range(-3, 0): # OPTION include -3 information
        for idx in range(-2, 0): 
            # 0 - 3 means it's outside bounds
            # 1 - 1 is inside bounds (predicting t=0)
            # 3 - 2 is inside bounds (predicting t=1 from t=3)
            sign = -1 if (t + idx < i) else 1
            
            try: 
                sc = -row[idx_to_key[idx]]
                if not np.isnan(sc):
                    score += sign * sc 
                    ideal += 1
            except KeyError:
                pass 
    
    return score / ideal  

def partition_doc(doc_scores, uni_concepts=True, length=50):
    # create and initialize the matrix 
    # doc = llama_scores.loc[llama_scores['doc_idx']==doc_idx]
    doc_unis = doc_scores.loc[doc_scores['n']==1].iloc[:length]
    n = len(doc_unis) 

    # implement as np array
    dp = np.ones((n, n))
    dp = dp * -1

    # fill out the matrix 
    for i in range(n):
        for j in range(n):
            if i <= j:
                if True: # j - i < 6:
                    dp[i][j] = psi(doc_unis, i, j)
    
    # look at heatmap of all the scores 
    # words = [w for w in doc_unis['decoded']]
    # sns.heatmap(dp, xticklabels=words, yticklabels=words)#, annot=True, fmt=".1g")
    # plt.show()

    # get the top scores in order 
    x, y = np.unravel_index(np.argsort(-dp.flatten()), dp.shape)
    coords = np.array(list(zip(x, y)))

    # go through all the top ngrams and add them to list, marking which ones become invalid as we go. 
    invalid = np.zeros_like(dp) + np.tril(np.ones_like(dp)) - np.eye(len(dp))
    segments = []
    for x, y in coords:
        condition = x <= y if uni_concepts else x < y
        if condition: 
            val = dp[x, y]
            
            if not invalid[x, y]:
                segments.append(((x,y), val))

                # strike out the cells that are now invalid 
                for i_x in range(invalid.shape[0]):
                    for i_y in range(invalid.shape[1]):
                        if i_y >= x and i_x <= y:
                            invalid[i_x, i_y] = 1
        
        if np.all(invalid):
            print("done, all invalid now")
            break 
    
    # if we were skipping unigrams above, we have to use them to fill in gaps 
    if not uni_concepts:
        for x, y in coords:
            if x == y:
                if not invalid[x, y]:
                    segments.append(((x,y), val))
                    invalid[x, y] = 1
        assert np.all(invalid)
    
    # validate that the segments fully cover doc 
    all_ranges = []
    for (x, y), val in segments:
        r = range(x, y+1)
        all_ranges += list(r)
    if set(all_ranges) == set(range(len(dp))):
        print("segments have full coverage")
    else:
        print("WARNING: segments did not fully cover doc")

    return segments

# take the segmentation of a document and read out the multi-token words 
# that were identified via this approach. 
def read_out_entries(segments, tokens, filter_unis=True):
    entries = []
    vals = []
    for (x, y), val in segments:
        r = range(x, y+1)
        if not (x==y and filter_unis):
            entries.append([tokens[idx] for idx in r])
            vals.append(val)
            
    return entries, vals

def get_tok_metrics(toks, i, start_probes, end_probes, start_states, end_states):
    corrstart, corrend, probdelta = {}, {}, {}
    
    # run each pair of probes on hidden states 
    for start_probe, end_probe, s in zip(start_probes, end_probes, tgt_idxs):
        label = {
            'i0' : toks[i],
            'in1' : toks[i - 1] if i >= 1 else None, 
            'in2' : toks[i - 2] if i >= 2 else None,
            'in3' : toks[i - 3] if i >= 3 else None
        }[s]
        
        if label is not None:
            start_logits = start_probe(start_states[i]).squeeze().detach().cpu()
            end_logits = end_probe(end_states[i]).squeeze().detach().cpu()
            
            corrstart[s] = start_logits.argmax() == label
            corrend[s] = end_logits.argmax() == label

            probdelta[s] = end_logits.softmax(dim=-1)[label].item() - start_logits.softmax(dim=-1)[label].item()

            del start_logits, end_logits

    return corrstart, corrend, probdelta 


# adapted from algodump.py. just get all the unigram information not ngrams.
# don't run it if it already exists 
def algodump(tokens, start_probes, end_probes, start_states, end_states, tokenizer):
    # per token: tok-1, decoded, tok-1_i0_iscorr, tok-1_i0_rankdelta, tok-1_i0_logitdelta, tok-1_i0_probdelta
    rows = []
    for i, ug_tok in enumerate(tokens):
        row = {'decoded' : tokenizer.decode(ug_tok), 'n' : 1}
        
        # for each token in the ng run all the relevant probes 
        corrstart, corrend, probdelta = \
            get_tok_metrics(tokens, i, start_probes, end_probes, start_states, end_states)
        
        # save this token
        row[f'tok-1'] = ug_tok
        
        # save i0, in1, in2, in3 for token-1 in the unigram
        for s in tgt_idxs:
            if s in corrstart.keys():
                row[f'tok-1_{s}_corrstart'] = corrstart[s].item()
                row[f'tok-1_{s}_corrend'] = corrend[s].item()
                row[f'tok-1_{s}_probdelta'] = probdelta[s]
        
        rows.append(row)

    del start_states, end_states
    torch.cuda.empty_cache()

    doc_df = pd.DataFrame(rows)
    return doc_df

def main(args):
    WINDOW_SIZE = 256
    MODEL_NAME = args.model.split('/')[-1]
    VOCAB_SIZE = {
        'Llama-2-7b-hf' : 32000,
        'Meta-Llama-3-8B' : 128256
    }[MODEL_NAME]
    MODEL_SIZE = {
        'vigogne-2-7b-instruct' : 4096,
        'meditron-7b' : 4096,
        'hf-llama-2' : 4096,
        'Llama-2-7b-hf' : 4096,
        'Llama-2-13b-hf' : 5120,
        'Llama-2-70b-hf' : 8192,
        'Meta-Llama-3-8B' : 4096
    }[MODEL_NAME]
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)    
    model = LanguageModel(args.model, device_map='cuda')

    # dataset we want to index 
    dataset = pd.read_csv(args.dataset)
    
    dump_dir = f"../logs/{MODEL_NAME}/candidates/{datasetname(args.dataset)}_layer{args.layer_start}_{args.layer_end}/"
    os.makedirs(dump_dir, exist_ok=True)
    
    if MODEL_NAME == "Llama-2-7b-hf":
        probes_csv = pd.read_csv(args.sweep)
    
        def get_probe(layer, target_idx):
            df = probes_csv.loc[(probes_csv['layer'] == layer) & (probes_csv['target_idx'] == target_idx)]
            assert(len(df)==1)
            
            ckpt = df.iloc[0]['Name']
            dir = "../checkpoints/hf-llama-2/{}/final.ckpt"
            
            probe = LinearModel(MODEL_SIZE, VOCAB_SIZE, bias=False).to('cuda')
            probe.load_state_dict(torch.load(dir.format(ckpt)))
            
            return probe    
    
    elif MODEL_NAME == "Meta-Llama-3-8B":
        def get_probe(layer, target_idx):
            search = f"Meta-Llama-3-8BLAYER{layer}-TGTIDX{target_idx}-train_tiny_1000-bsz4-lr0.10000-epochs16"
            # dir = "/data/david_atkinson/lexicon/checkpoints/Meta-Llama-3-8B/meta-llama/"
            dir = "../checkpoints/Meta-Llama-3-8B/meta-llama/"
            
            for fname in os.listdir(dir):
                if search in fname:
                    dir += fname + '/final.ckpt'
            
            print(dir)
            probe = LinearModel(MODEL_SIZE, VOCAB_SIZE, bias=False).to('cuda')
            probe.load_state_dict(torch.load(dir))
            return probe 

    else:
        raise Exception(f"don't have probes for {MODEL_NAME}") 

    # load in the probes at layer_start and layer_end 
    start_probes, end_probes = [], []
    start_probes.append(get_probe(args.layer_start, 0))
    end_probes.append(get_probe(args.layer_end, 0))

    start_probes.append(get_probe(args.layer_start, -1))
    end_probes.append(get_probe(args.layer_end, -1))

    start_probes.append(get_probe(args.layer_start, -2))
    end_probes.append(get_probe(args.layer_end, -2))
    
    ctr = 0
    all_ctr = Counter()
    sum_scores = {}
    col = 'text' if 'wikipedia' in args.dataset else 'decoded_prefix'
    tik = time.time()
    for doc_idx, doc in enumerate(dataset[col]):
        # get tokens and states
        tokens = tokenizer(doc)['input_ids'][:WINDOW_SIZE]

        with torch.no_grad():
            with model.trace(tokens):
                ss = model.model.layers[args.layer_start].output[0].squeeze().save()
                es = model.model.layers[args.layer_end].output[0].squeeze().save()

            start_states = ss.detach()
            end_states = es.detach()
            del ss, es 
        
        # get probe probability information for this doc_idx
        fname = f"algodump_{doc_idx}.csv"
        try: 
            doc_df = pd.read_csv(dump_dir + fname)
            print(f"loaded {dump_dir + fname}")
        except FileNotFoundError:
            doc_df = algodump(tokens, start_probes, end_probes, start_states, end_states, tokenizer)
            doc_df.to_csv(dump_dir + fname, quoting=csv.QUOTE_ALL)
            print(f"saved {dump_dir + fname}, {len(tokens)} tokens in {datasetname(args.dataset)}")
        
        # THE RECIPE: use uni_concepts=True to partition document
        picklename = f"segments_{doc_idx}.pkl"
        try:
            with open(dump_dir + picklename, 'rb') as f:
                segments = pickle.load(f)
            print(f"loaded segments from {dump_dir + picklename}")
            
        except FileNotFoundError:
            print(f"partitioning doc {doc_idx}...")
            segments = partition_doc(doc_df, uni_concepts=True, length=len(tokens))
            
            with open(dump_dir + picklename, 'wb') as f:
                pickle.dump(segments, f)
            print(f"saved segments to {dump_dir + picklename}")
        
        # but filter out the unigrams when you're "reading things out"
        entries, vals = read_out_entries(segments, tokens, filter_unis=True)
        decoded_entries = [tokenizer.decode(e) for e in entries]
        
        # add to running totals
        all_ctr += Counter(decoded_entries)
        for de, v in zip(decoded_entries, vals):
            try: 
                sum_scores[de] += v
            except KeyError:
                sum_scores[de] = v
    
        ctr += 1
        if args.n_examples > 0: 
            if ctr >= args.n_examples:
                break # cut off
    
    tok = time.time()
    print("minutes taken:", (tok-tik) / 60)

    assert all_ctr.keys() == sum_scores.keys()
    
    # save counts of all vocabulary items 
    cts_fname = f"cts_0thru{doc_idx}.pkl"
    with open(dump_dir + cts_fname, 'wb') as f:
        pickle.dump(all_ctr, f)
    print(f"saved counts at {dump_dir + cts_fname}")
    
    # calculate average scores for each vocab item
    avg_scores = {}
    for k, v in sum_scores.items():
        avg_scores[k] = v / all_ctr[k]

    # save average scores for each vocab item
    avgs_fname = f"avgs_0thru{doc_idx}.pkl"
    with open(dump_dir + avgs_fname, 'wb') as f:
        pickle.dump(avg_scores, f)
    print(f"saved averages at {dump_dir + avgs_fname}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", default="../data/test_tiny_500.csv")
    parser.add_argument('--layer_start', type=int, default=1)
    parser.add_argument('--layer_end', type=int, default=9)
    parser.add_argument('--n_examples', type=int, default=-1)
    
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--sweep', default='../sweeps/finalsweep.csv')
    
    args = parser.parse_args()
    
    main(args)
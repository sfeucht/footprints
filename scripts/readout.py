'''
Given a dataset csv with the column 'text' and a choice between Llama-2-7b and
Llama-3-8b, "read out" the vocabulary of that model based on the given datset.
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
from training import LinearModel
from nnsight import LanguageModel
from huggingface_hub import hf_hub_download

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

def get_probe(layer, target_idx, model):
    # example: llama-2-7b probe at layer 0, predicting 3 tokens ago
    # predicting the next token would be `layer0_tgtidx1.ckpt`
    checkpoint_path = hf_hub_download(
        repo_id="sfeucht/footprints",
        filename=f"{model}/layer{layer}_tgtidx{target_idx}.ckpt"
    )

    # model_size is 4096 for both models.
    # vocab_size is 32000 for Llama-2-7b and 128256 for Llama-3-8b
    model_size = 4096
    if model == 'llama-2-7b':
        vocab_size = 32000
    elif model == 'llama-3-8b':
        vocab_size = 128256

    probe = LinearModel(model_size, vocab_size).cuda()
    probe.load_state_dict(torch.load(checkpoint_path))

    return probe


'''
Implementation of "Erasure Score" 
'''
def psi(doc_info, i, j):
    sn = doc_info.iloc[i:j+1]

    idx_to_key = {
        0 : 'tok-1_i0_probdelta',
        -1 : 'tok-1_in1_probdelta',
        -2 : 'tok-1_in2_probdelta',
        -3 : 'tok-1_in3_probdelta'
    }

    # we're doing 0 indexing for t, so this is different from the paper
    # in the paper we did start-end, so we have to flip all these to end-start and add - in front of probdelta
    # also, we have to do (t + idx < i) since we're using absolute idxs

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

'''
Given doc_info, apply Algorithm 1 to segment this particular document into non-
overlapping high-scoring chunks. Allow for unigram segments to fill in the gaps.
'''
def partition_doc(doc_info):
    # create and initialize the matrix
    n = len(doc_info)

    # implement as np array
    dp = np.ones((n, n))
    dp = dp * -1

    # fill out the matrix
    for i in range(n):
        for j in range(n):
            if i <= j:
                if True: # j - i < 6:
                    dp[i][j] = psi(doc_info, i, j)

    # get the top scores in order
    x, y = np.unravel_index(np.argsort(-dp.flatten()), dp.shape)
    coords = np.array(list(zip(x, y)))

    # go through all the top ngrams and add them to list, marking which ones become invalid as we go.
    segments = []
    for p, q in coords:
        if p <= q:
            val = dp[p, q]

            valid = True
            for (x, y), _ in segments:
              if x > q or y < p:
                pass
              else:
                valid = False
                break

            if valid:
                segments.append(((p,q), val))

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

'''
Run all the possible probes for a specific token 
'''
def get_tok_metrics(toks, i, start_probes, end_probes, start_states, end_states, tgt_idxs):
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


'''
given a bunch of tokens and the states for the tokens at a layer, create a dataframe
with every possible probdelta (for different target indices) for each token.
'''
def get_doc_info(tokens, model, layer_start, layer_end, start_probes, end_probes, tokenizer):
  tgt_idxs = ['i0', 'in1', 'in2', 'in3']

  # get hidden states for tokens
  with torch.no_grad():
    with model.trace(tokens):
        ss = model.model.layers[layer_start].output[0].squeeze().save()
        es = model.model.layers[layer_end].output[0].squeeze().save()

    start_states = ss.detach()
    end_states = es.detach()
    del ss, es

  # per token: tok-1, decoded, tok-1_i0_iscorr, tok-1_i0_rankdelta, tok-1_i0_logitdelta, tok-1_i0_probdelta
  rows = []
  for i, ug_tok in enumerate(tokens):
      row = {'decoded' : tokenizer.decode(ug_tok), 'n' : 1}

      # for each token in the ng run all the relevant probes
      corrstart, corrend, probdelta = \
          get_tok_metrics(tokens, i, start_probes, end_probes, start_states, end_states, tgt_idxs)

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

  return pd.DataFrame(rows)

def main(args):
    model = LanguageModel(args.model, device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.model)    
    
    MODEL_NAME = args.model.split('/')[-1]
    WINDOW_SIZE = 256
    
    # dataset we want to index 
    dataset = pd.read_csv(args.dataset)
    
    dump_dir = f"../logs/{MODEL_NAME}/readout/{datasetname(args.dataset)}_layer{args.layer_start}_{args.layer_end}/"
    os.makedirs(dump_dir, exist_ok=True)
    
    hf_string = {
        'Llama-2-7b-hf' : 'llama-2-7b',
        'Meta-Llama-3-8B' : 'llama-3-8b'
    }[MODEL_NAME]
    
    # load in the probes at layer_start and layer_end 
    start_probes, end_probes = [], []
    start_probes.append(get_probe(args.layer_start, 0, hf_string))
    end_probes.append(get_probe(args.layer_end, 0, hf_string))

    start_probes.append(get_probe(args.layer_start, -1, hf_string))
    end_probes.append(get_probe(args.layer_end, -1, hf_string))

    start_probes.append(get_probe(args.layer_start, -2, hf_string))
    end_probes.append(get_probe(args.layer_end, -2, hf_string))
    
    ctr = 0
    all_ctr = Counter()
    sum_scores = {}
    tik = time.time()
    for doc_idx, doc in enumerate(dataset['text']):
        tokens = tokenizer(doc)['input_ids'][:WINDOW_SIZE]
        
        # get probe probability information for this doc_idx
        fname = f"docinfo_{doc_idx}.csv"
        try: 
            doc_df = pd.read_csv(dump_dir + fname)
            print(f"loaded {dump_dir + fname}")
        except FileNotFoundError:
            doc_df = get_doc_info(tokens, model, args.layer_start, args.layer_end, start_probes, end_probes, tokenizer)
            doc_df.to_csv(dump_dir + fname, quoting=csv.QUOTE_ALL)
            print(f"saved {dump_dir + fname}, {len(tokens)} tokens in {datasetname(args.dataset)}")
        
        # segment doc with partition_doc 
        picklename = f"segments_{doc_idx}.pkl"
        try:
            with open(dump_dir + picklename, 'rb') as f:
                segments = pickle.load(f)
            print(f"loaded segments from {dump_dir + picklename}")
            
        except FileNotFoundError:
            print(f"partitioning doc {doc_idx}...")
            segments = partition_doc(doc_df)
            
            with open(dump_dir + picklename, 'wb') as f:
                pickle.dump(segments, f)
            print(f"saved segments to {dump_dir + picklename}")
        
        # filter out the unigrams when you're "reading out" the vocabulary 
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
    
    ctr = 0
    print("\nTop 50 Vocabulary Entries")
    for k, v in avg_scores.items():
        print(repr(k), '\t', v)
        ctr += 1
        if ctr > 50:
            break 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", default="../data/test_tiny_500.csv")
    parser.add_argument('--layer_start', type=int, default=1)
    parser.add_argument('--layer_end', type=int, default=9)
    parser.add_argument('--n_examples', type=int, default=-1, help="-1 to use the whole dataset")
    
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf', 
                        choices=['meta-llama/Meta-Llama-3-8B', 'meta-llama/Llama-2-7b-hf'])
    
    args = parser.parse_args()
    
    main(args)
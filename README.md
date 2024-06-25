# Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs
How do LLMs process multi-token words, common phrases, and named entities? We discover a pattern of token erasure that we hypothesize to be a 'footprint' of how LLMs process unnatural tokenization. Read about our paper here: https://footprints.baulab.info

<img src="https://github.com/sfeucht/footprints/assets/56804258/78d7d86b-81e7-4818-8521-0c05e05934f2" width="500" />

## Setup
Create a virtual environment with Python 3.8.10 and install the required packages.
```
python -m venv env
pip install -r requirements.txt
```

## Demo
Checkpoints for each of the linear probes used in our paper are available at https://huggingface.co/sfeucht/footprints. (todo explain)

## Code
Code used in this paper for training and testing linear probes can be found in `./scripts`. We have provided the probes used for the paper above. However, if you would still like to train your own linear probe on e.g. layer 12 to predict the previous two tokens, run
```
python train_probe.py --layer 12 --target_idx -2 
```
and a linear model will be trained on Llama-2-7b by default and stored as a checkpoint in `./checkpoints`. These checkpoints can then be read by `./scripts/test_probe.py` and tested on either CounterFact tokens, Wikipedia tokens (multi-token words or spaCy entities), or plain Pile tokens. Test results are stored in `./logs`. 
```
python test_probe.py --checkpoint ../checkpoints/Llama-2-7b-hf/.../final.ckpt --test_data counterfact_expanded.csv
```

## Data
We use three datasets in this paper. 

- CounterFact:
    - expanded1_text (rename) used for all of the CounterFact tests in the paper
- Pile:
    - train_tiny_1000.csv 'text' column used to train probes
    - val_tiny_500.csv used to validate probe hparams
    - test_tiny_500.csv used for overall Pile test results
- Wikipedia:
    - wikipedia_val_500.csv (rename) used for overall Wikipedia test results
    - wikipedia_test_500 and wikipedia_train_1000.csv untouched, kept for posterity

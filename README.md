# Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs
How do LLMs process multi-token words, common phrases, and named entities? We discover a pattern of token erasure that we hypothesize to be a 'footprint' of how LLMs process unnatural tokenization. Read about our paper here: https://footprints.baulab.info

<img src="https://github.com/sfeucht/footprints/assets/56804258/78d7d86b-81e7-4818-8521-0c05e05934f2" width="500" />

## Demo: Segmenting a Document
To see the *erasure score* from our paper in action, check out our [demo](), which allows you to run our probes on any chunk of text to view the highest-scoring multi-token lexical items. This demo applies the same procedure that was used to segment the document below (and the examples in the paper appendix). 

<img width="500" alt="Monk example from website" src="https://github.com/sfeucht/footprints/assets/56804258/5ba3c7dd-da0b-4b2b-9a91-be86bdb0afb6">

## "Reading Out" Vocabulary Examples
To run this algorithm for a large number of documents in order to inspect the top-scoring token sequences, first clone this repository and create a new virtual environment using Python 3.8.10:
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
Then, you can run 
```
python readout.py ... 
```
to apply Algorithm 1 from our paper to every document in .... Outputs are stored in two dictionaries: `` and ``. 

## Loading Our Probes
Checkpoints for each of the linear probes used in our paper are available at https://huggingface.co/sfeucht/footprints. To load a linear probe used in this paper, run the following code snippet:

```python
import torch 
import torch.nn as nn
from huggingface_hub import hf_hub_download

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size, bias=False):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=bias)
    def forward(self, x):
        output = self.fc(x)
        return output

# example: llama-2-7b probe at layer 0, predicting 3 tokens ago
# predicting the next token would be `layer0_tgtidx1.ckpt`
checkpoint_path = hf_hub_download(
    repo_id="sfeucht/footprints", 
    filename="llama-2-7b/layer0_tgtidx-3.ckpt"
)

# model_size is 4096 for both models.
# vocab_size is 32000 for Llama-2-7b and 128256 for Llama-3-8b
probe = LinearModel(4096, 32000).cuda()
probe.load_state_dict(torch.load(checkpoint_path))
```

## Training Your Own Probes
Code used in this paper for training and testing linear probes can be found in `./scripts`. We have provided the probes used for the paper above. However, if you would still like to train your own linear probe on e.g. layer 12 to predict the previous two tokens, run
```
python train_probe.py --layer 12 --target_idx -2 
```
and a linear model will be trained on Llama-2-7b by default and stored as a checkpoint in `./checkpoints`. These checkpoints can then be read by `./scripts/test_probe.py` and tested on either CounterFact tokens, Wikipedia tokens (multi-token words or spaCy entities), or plain Pile tokens. Test results are stored in `./logs`. 
```
python test_probe.py --checkpoint ../checkpoints/Llama-2-7b-hf/.../final.ckpt --test_data counterfact_expanded.csv
```

## Datasets Used
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

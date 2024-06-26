# Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs
How do LLMs process multi-token words, common phrases, and named entities? We discover a pattern of token erasure that we hypothesize to be a 'footprint' of how LLMs process unnatural tokenization. Read about our paper here: https://footprints.baulab.info

<img src="https://github.com/sfeucht/footprints/assets/56804258/78d7d86b-81e7-4818-8521-0c05e05934f2" width="500" />

## Demo: Segmenting a Document
To see the *erasure score* from our paper in action, check out our [demo](), which allows you to run our probes on any chunk of text to view the highest-scoring multi-token lexical items. This colab notebook implements the same procedure that was used to segment the document below, as well as the examples in the paper appendix.

<img width="500" alt="Monk example from website" src="https://github.com/sfeucht/footprints/assets/56804258/5ba3c7dd-da0b-4b2b-9a91-be86bdb0afb6">

## "Reading Out" Vocabulary Examples
To run this algorithm for a large number of documents in order to inspect the top-scoring token sequences, first clone this repository and create a new virtual environment using Python 3.8.10:
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
Then, you can run e.g.
```
python readout.py --n_examples 2 --model meta-llama/Meta-Llama-3-8B --dataset ../data/wikipedia_test_500.csv
```
to replicate Appendix Table 6. We also provide a script version of the above demo as `segments.py`, which allows you to input a document as a txt file and view the resulting multi-token sequences (optionally in html format with the `--output_html` flag). 

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
We use three datasets in this paper, which can all be found in `./data`. 

- CounterFact [(Meng et al., 2022)](https://rome.baulab.info/)
    - `counterfact_expanded.csv` was used for all of the CounterFact tests in the paper, and includes rows in addition to the original CounterFact dataset.
- Pile [(Gao et al., 2020)](https://pile.eleuther.ai/)
    - `train_tiny_1000.csv` was used to train all of the probes. 
    - `val_tiny_500.csv` was used to validate probe hyperparameters.
    - `test_tiny_500.csv` was used for overall Pile test results.
- Wikipedia [(Wikimedia Foundation, 2022)](https://huggingface.co/datasets/legacy-datasets/wikipedia)
    - `wikipedia_test_500.csv` was used for overall Wikipedia test results.
    - `wikipedia_val_500.csv` and `wikipedia_train_1000.csv` were not used in this work, but are included for completeness. 

# Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs
How do LLMs process multi-token words, common phrases, and named entities? We discover a pattern of token erasure that we hypothesize to be a 'footprint' of how LLMs process unnatural tokenization. 

Read more about our paper here: <br>
🌐 https://footprints.baulab.info <br>
📄 https://arxiv.org/abs/2406.20086

<img src="https://github.com/sfeucht/footprints/assets/56804258/78d7d86b-81e7-4818-8521-0c05e05934f2" width="500" />

## Setup 
To run our code, clone this repository and create a new virtual environment using Python 3.8.10:
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Segmenting a Document
An implementation of Algorithm 1 in our paper is provided in `segment.py`. This script can be run like so:
```
python segment.py --document my_doc.txt --model meta-llama/Llama-2-7b-hf
```
allowing you to segment any paragraph of text into high-scoring token sequences.
```
segments from highest to lowest score:
'dramatic'       0.5815845847818102
'twists'         0.5553912909024803
'low bass'       0.41476866118921824
'cuss'   0.3979072428604316
'fifth'          0.3842911866668146
'using'          0.3568337553491195
...
'ive'    -0.07994025301498671
's'      -0.14006704260206485
'ations'         -0.2306471753618856
'itions'         -0.3348596893891435
```
Adding the `--output_html` flag will also save an HTML file in the style of the below example to the folder `./logs/html`, bolding all multi-token sequences and coloring them blue if they have a higher erasure score.

<img width="700" alt="Monk example from website" src="https://github.com/sfeucht/footprints/assets/56804258/5ba3c7dd-da0b-4b2b-9a91-be86bdb0afb6">

## "Reading Out" Vocabulary Examples
To apply this segmentation algorithm to an entire dataset (as seen in Tables 3 through 6), run
```
python readout.py --model meta-llama/Meta-Llama-3-8B --dataset ../data/wikipedia_test_500.csv
```
which specifically replicates Appendix Table 6. You can use your own dataset csv, as long as it contains a 'text' column with the documents you want to analyze.  

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
We have provided the probes used for the paper above. However, if you would still like to train your own linear probes, we provide code for training and testing linear probes on Llama hidden states in `./scripts`. To train a probe on e.g. layer 12 to predict two tokens ago, run
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

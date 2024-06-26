'''
Train a new linear probe to predict a token offset (target_idx) at a specific 
layer. For example, you can train a probe with target_idx=-2 and layer=12 to 
take hidden states at layer 12 (the 13th layer) and predict what was two tokens
before that hidden state.
'''
import os
import csv
import wandb
import torch
import argparse

import pandas as pd
import torch.nn.functional as F
import pytorch_warmup as warmup

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nnsight import LanguageModel

from training import LinearModel, train_epoch, test, DocDataset, DocCollate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.login()

def datasetname(input):
    return input.split('/')[-1][:-4]

def add_args(s, args):
    for k, v in vars(args).items():
        if k in ['probe_bsz', 'probe_epochs']:
            s += f"-{k[6:]}{v}"
        elif k in ['probe_lr']:
            s += f"-{k[6:]}" + "{:1.5f}".format(v)
    return s

def main(args):
    model = LanguageModel(args.model, device_map=device)
    tokenizer = model.tokenizer 
    tokenizer.add_special_tokens({'pad_token':'<s>'})

    VOCAB_SIZE = model.vocab_size
    MODEL_SIZE = model.config.hidden_size
    MODEL_NAME = args.model.split('/')[-1]

    # max_seq_len from below sources is 2048, but changing to 512 for memory/speed 
    # https://github.com/meta-llama/llama/blob/main/llama/model.py#L31
    # https://github.com/meta-llama/llama3/blob/bf8d18cd087a4a0b3f61075b7de0b86cf6c70697/llama/model.py#L32
    WINDOW_SIZE = 512 

    for param in model.parameters():
        param.requires_grad = False 
        
    run_name = add_args(f"{MODEL_NAME}LAYER{args.layer}-TGTIDX{args.target_idx}-{datasetname(args.train_data)}", args)
    wandb.init(project = args.wandb_proj, name = run_name, config = args, settings=wandb.Settings(start_method="fork"))
    
    run_name += f'-{wandb.run.id}'
    wandb.run.name = run_name
    
    if args.probe_bsz > 10:
        print(f"Warning: batch size represents number of documents (each doc contains a few hundred tokens). {args.probe_bsz}>10, you may want to use a smaller batch size.")

    # make dirs that include the wandb id
    checkpoint_dir = f"../checkpoints/{MODEL_NAME}/{run_name}"
    log_dir = f"../logs/{MODEL_NAME}/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # load data csvs 
    train_data = pd.read_csv(args.train_data)
    val_data = pd.read_csv(args.val_data)
    test_data = pd.read_csv(args.test_data)

    # pass in subjects from counterfact dataset as "entities" to split during testing 
    which_entity = "subject"
    entities = None
    if test_data is not None: 
        if which_entity in test_data.columns:
            entities = list(test_data[which_entity])
    
    train_dataset = DocDataset(model, tokenizer, args.layer, args.target_idx, train_data, WINDOW_SIZE, VOCAB_SIZE, device)
    val_dataset = DocDataset(model, tokenizer, args.layer, args.target_idx, val_data, WINDOW_SIZE, VOCAB_SIZE, device)
    test_dataset = DocDataset(model, tokenizer, args.layer, args.target_idx, test_data, WINDOW_SIZE, VOCAB_SIZE, device, entities=entities)
    
    linear_probe = LinearModel(MODEL_SIZE, VOCAB_SIZE).to(device)
    wandb.watch(linear_probe)

    collate_fn = DocCollate(args.layer, args.target_idx, tokenizer, model, WINDOW_SIZE, device)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.probe_bsz, collate_fn=collate_fn, 
        drop_last=True, pin_memory=False, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.probe_bsz, collate_fn=collate_fn, 
        drop_last=True, pin_memory=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.probe_bsz, collate_fn=collate_fn, 
        drop_last=True, pin_memory=False)

    optimizer = torch.optim.AdamW(linear_probe.parameters(), lr=args.probe_lr, weight_decay=args.probe_wd) # no momentum
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    
    criterion = {
        "ce" : F.cross_entropy,
        "mse" : F.mse_loss
    }[args.criterion]
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

    print('training linear probe...') 
    batches_seen = 0
    for epoch in range(args.probe_epochs):
        print('# Epoch {} #'.format(epoch))
        batches_seen = train_epoch(epoch, linear_probe, train_loader, criterion, optimizer, warmup_scheduler, args.accumulate, args.clip_threshold, batches_seen, device)

        # log validation loss at the end of each epoch
        val_loss, val_acc, val_topk_acc, val_entity_ng_acc, val_other_acc = test(linear_probe, val_loader, criterion, device, model, tokenizer, args.target_idx, return_results=False)
        wandb.log({"val_loss": val_loss, "val_acc": val_acc, "val_topk_acc": val_topk_acc, "val_entity_ng_acc": val_entity_ng_acc, "val_other_acc":val_other_acc})

        if warmup_scheduler is not None:
            with warmup_scheduler.dampening():
                scheduler.step(val_loss)
        else:
            scheduler.step(val_loss)
    
    # Get final testing accuracy and prediction results
    torch.save(linear_probe.state_dict(), f"{checkpoint_dir}/final.ckpt") 
    test_loss, test_acc, test_topk_acc, test_entity_ng_acc, test_other_acc, test_results = test(linear_probe, test_loader, criterion, device, model, tokenizer, args.target_idx, return_results=True)
    test_results.to_csv(log_dir + f"/{datasetname(args.test_data)}_results.csv", quoting=csv.QUOTE_ALL, encoding='utf-8')
    
    print('Test Loss: {:10.4f}  Accuracy: {:3.4f}%\n'.format(test_loss, test_acc))
    wandb.log({"test_loss": test_loss, "test_acc": test_acc, "test_topk_acc": test_topk_acc, "test_entity_ng_acc": test_entity_ng_acc, "test_other_acc": test_other_acc})
    wandb.finish()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training info for linear probe
    parser.add_argument('--probe_bsz', type=int, default=1)
    parser.add_argument('--probe_lr', type=float, default=0.1)
    parser.add_argument('--probe_wd', type=float, default=0.001)
    parser.add_argument('--probe_epochs', type=int, default=8)

    parser.add_argument('--wandb_proj', type=str, default='footprints')
    parser.add_argument('--accumulate', type=int, default=30)
    parser.add_argument('--clip_threshold', type=float, default=0.1)

    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--criterion', type=str, choices=['mse', 'ce'], default='ce')

    # document data locations
    parser.add_argument('--train_data', type=str, default='../data/train_tiny_1000.csv')
    parser.add_argument('--val_data', type=str, default='../data/val_tiny_500.csv')
    parser.add_argument('--test_data', type=str, default='../data/test_tiny_500.csv')

    # required specifications for where probe is trained 
    parser.add_argument('--layer', type=int, required=True,
                        help='which layer to train the probe at, from -1...32 where -1 is embedding layer and 32 is output pre-softmax.')
    parser.add_argument('--target_idx', type=int, required=True,
                        help='which token the probe should predict from current hidden state (e.g. 0 for current token, -1 for prev)')
    parser.add_argument('--model', type=str, choices=['meta-llama/Llama-2-7b-hf', 'meta-llama/Meta-Llama-3-8B'], default='meta-llama/Llama-2-7b-hf')


    args = parser.parse_args()
    main(args)
    
    

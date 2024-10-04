'''
Segment one document into highest-scoring token subsequences.
'''
import os
import argparse
import math
from nnsight import LanguageModel
from readout import get_doc_info, partition_doc, get_probe

## HTML Formatting
def col(s):
    # make sure between -1 and 1 
    s = max(-1, min(1, s))
    
    # blue (high) 0, 98, 255
    blue_r = 0
    blue_g = 98
    blue_b = 255

    if (s > 0):
        r = math.floor(255 + s * (blue_r - 255))
        g = math.floor(255 + s * (blue_g - 255))
        b = math.floor(255 + s * (blue_b - 255))

        return f"style='background-color:rgb({r}, {g}, {b})'"
    else:
        return ""

def html_span(segments, doc_unis, tokenizer):
    # all the tokens in original doc
    doc_tokens = [t for t in doc_unis['tok-1']]

    boxs_col = lambda x: tokenizer(f"<span title='{x}' {col(x)}>")['input_ids'][1:]
    boxs = lambda x: tokenizer(f"<span title='{x}'>")['input_ids'][1:]
    boxe = tokenizer("</span>")['input_ids'][1:]
    
    blds = tokenizer("<b>")['input_ids'][1:]
    blde = tokenizer("</b>")['input_ids'][1:]

    bos_tok = tokenizer("&lt;s&gt;")['input_ids'][1:]

    out = []
    for (x, y), val in segments:
        tokens = doc_tokens[x:y+1]
        
        # replace <s> with s 
        if tokens[0] == 1:
            tokens = bos_tok + tokens[1:] # 's'  
            
        if len(tokens) > 1:          
            to_appnd = [*boxs_col(val), *blds, *tokens, *blde, *boxe]
        else:
            to_appnd = [*boxs(val), *tokens, *boxe]

        out += to_appnd

    return tokenizer.decode(out)

def html_view(segments, doc_unis, tokenizer):
    # sort segments to be in document order 
    segments = sorted(segments, key=lambda t: t[0][0])
    inject = html_span(segments, doc_unis, tokenizer)
    
    # white out non-mtw segment values 
    whiteout = []
    for (x,y), val in segments:
        if x == y:
            whiteout.append(((x,y), 0))
        else:
            whiteout.append(((x,y), val))

    template = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        #text-container span {{
        transition: background-color 0.3s;
        border: rgb(105, 102, 102) dotted;
        padding: 0px;
        cursor: default;
        }}
    </style>
    </head>
    <body>
    
    <p id="text-container">
        {inject}
    </p>
    
    </body>
    </html>
    """

    return template


def load_model_and_probes(path):
    model = LanguageModel(path, device_map='cuda')
    tokenizer = model.tokenizer
    
    hf_name = { 
        'meta-llama/Llama-2-7b-hf' : 'llama-2-7b', 
        'meta-llama/Meta-Llama-3-8B' : 'llama-3-8b'
    }[path]
    layer_start = 1
    layer_end = 9

    start_probes, end_probes = [], []
    start_probes.append(get_probe(layer_start, 0, hf_name))
    end_probes.append(get_probe(layer_end, 0, hf_name))

    start_probes.append(get_probe(layer_start, -1, hf_name))
    end_probes.append(get_probe(layer_end, -1, hf_name))

    start_probes.append(get_probe(layer_start, -2, hf_name))
    end_probes.append(get_probe(layer_end, -2, hf_name))
    
    return model, tokenizer, start_probes, end_probes


def main(args):
    # load in model and probes 
    model, tokenizer, start_probes, end_probes = load_model_and_probes(args.model)
    
    # read in given txt file 
    with open(args.document, 'r') as f:
        input_text = f.read().strip()
    
    # tokenize 
    tokens = tokenizer(input_text)['input_ids'][:args.max_length]
    
    # get probe info and partition document 
    doc_info = get_doc_info(tokens, model, args.layer_start, args.layer_end, start_probes, end_probes, tokenizer)
    segments = partition_doc(doc_info)
    
    # save html output if desired
    if args.output_html:
        html_output = html_view(segments, doc_info, tokenizer)
    
        write_dir = "../logs/html/"
        fname = f"{args.document.split('/')[-1][:-4]}.html"
        os.makedirs(write_dir, exist_ok=True)
    
        print("saving html output as " + write_dir + fname)
        with open(f"../logs/html/{args.document.split('/')[-1][:-4]}.html", 'w') as f:
            f.write(html_output)
    
    # print document segments 
    print("\nsegments from highest to lowest score:")
    for (p, q), val in segments:
        text = tokenizer.decode(tokens[p : q+1])
        print(repr(text), '\t', val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--document", default="../data/monk.txt")
    parser.add_argument('--layer_start', type=int, default=1)
    parser.add_argument('--layer_end', type=int, default=9)
    parser.add_argument('--max_length', type=int, default=256)
    
    parser.add_argument('--output_html', action='store_true')
    parser.set_defaults(output_html=False)
    
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf', 
                        choices=['meta-llama/Meta-Llama-3-8B', 'meta-llama/Llama-2-7b-hf'])
    
    args = parser.parse_args()
    
    main(args)

"""
Author: Eric DeMattos
Generates custom embeddings (token + speech features) for each data split.
"""

import pdb
import os
import torch

def save(token_feats, speech_feats, emb_idxs, glove_vocab):
    """
    Append the current batch of embeddings to file.
    """

    start_idx = glove_vocab['<START>']  # 400000
    stop_idx = glove_vocab['<STOP>']    # 400001
    unk_idx = glove_vocab['<unk>']      # 400002

    # invert dictionary (word: idx -> idx: word)
    glove_vocab = {v: k for k, v in glove_vocab.items()}

    # join token embeddings with prosody embeddings along axis 1
    embed_concat = torch.cat([token_feats, speech_feats], 1)

    # hardcode paths for now
    proj_home = '/home/s2057915/prosody_nlp/code/self_attn_speech_parser'
    data_split = 'turn_dev'

    # load all tokens for this split and assign each a unique ID
    with open(f"{proj_home}/stanza_data/{data_split}_indices.txt") as f:
        tokens = [f"{token.strip()}+{data_split}+{i}" for i, token in enumerate(f)]

    ### store current batch in temp dictionary
    ### embed_dict = dict()

    ### collect number of <START> and <STOP> tokens to subtract from total
    ### num_meta_token = 0

    # write current batch to file
    with open(f"{proj_home}/{data_split}_embed.txt", "a") as f:

        for i, idx in enumerate(emb_idxs):
    
            token = glove_vocab[int(idx)]

            # meta tokens have the same embedding; no prosodic difference
            if token == '<START>' or token == '<STOP>':
                print(f"skipping: {token} {' '.join([str(x) for x in embed_concat[i].tolist()[:5]])}")
                continue

            # example line: "aardvark+turn_dev+3235 -0.5 2.5 [...] 1.2 -0.7"
            f.write(f"{tokens[i]} {' '.join([str(x) for x in embed_concat[i].tolist()])}\n")


### remove duplicate meta tokens from total count
### vocab_size = len(tokens) - num_meta_token + 2  # add 1 for <START> and 1 for <STOP>

    print(f"len(tokens) = {len(tokens)}\nglove_vocab = {len(glove_vocab.keys())}")

### word2vec/fastText format requires vocabulary size and embed dimension first
### f.write(f"{vocab_size} 1024\n")

### get embedding for meta tokens
### f.write(f"<START> {' '.join([str(x) for x in embed_concat[emb_idxs.tolist().index(start_idx)]])}\n")
### f.write(f"<STOP> {' '.join([str(x) for x in embed_concat[emb_idxs.tolist().index(stop_idx)]])}")

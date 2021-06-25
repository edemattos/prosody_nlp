"""
Author: Eric DeMattos
Generates custom embeddings (token + speech features) for each data split.
"""

import pdb
import os
import torch

def save(token_feats, speech_feats, emb_idxs, glove_vocab):
    """
    Append the current batch of embeddings to file; ensure batch size contains ALL instances!
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
    data_split = 'turn_dev'  # aka ${PREFIX}

    # load all tokens for this split and assign each a unique ID
    with open(f"{proj_home}/stanza_data/{data_split}_indices.txt") as f:
       tokens = [f"{token.strip()}+{data_split}+{i}" for i, token in enumerate(f)]

    # append current batch; ensure file does not exist before running!
    with open(f"{proj_home}/{data_split}_embed.txt", "a") as f:

        # TODO: move this to post processing
        # get embedding for meta tokens
        # f.write(f"<START> {' '.join([str(float(x)) for x in embed_concat[emb_idxs.tolist().index(start_idx)]])}\n")
        # f.write(f"<STOP> {' '.join([str(float(x)) for x in embed_concat[emb_idxs.tolist().index(stop_idx)]])}\n")

        for i, idx in enumerate(emb_idxs):

            # meta tokens have the same embedding; no prosodic difference
            if glove_vocab[int(idx)] == '<START>' or glove_vocab[int(idx)] == '<STOP>':
                # print(f"skipping: {glove_vocab[int(idx)]} {' '.join([str(x) for x in embed_concat[i].tolist()[:5]])}")
                continue

            # example: "aardvark+turn_dev+3235 -0.5 1.2 -0.7"
            # f.write(f"{tokens[i]} {' '.join([str(x) for x in embed_concat[i].tolist()])}\n")
            f.write(f"{glove_vocab[int(idx)]}+{data_split}+{i} {' '.join([str(x) for x in embed_concat[i].tolist()])}\n")

    # post processing: add vocab length and embed dimension to top of file
    # wc -l turn_dev_embed.txt >> turn_dev_embed_fixed.txt
    # cat turn_dev_embed.txt >> turn_dev_embed_fixed.txt
    # rm turn_dev_embed.txt

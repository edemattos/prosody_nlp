"""
Author: Eric DeMattos
Stores all custom embeddings (token + speech features) for each data split.
"""

import os
import torch

def save(token_feats, speech_feats, emb_idxs, glove_vocab, sentences, sent_ids):

    # invert dictionary (word: idx -> idx: word)
    glove_vocab = {v: k for k, v in glove_vocab.items()}

    # join token embeddings with prosody embeddings along axis 1
    embed_concat = torch.cat([token_feats, speech_feats], 1)

    # hardcode paths for now
    proj_home = '/home/s2057915/prosody_nlp/code/self_attn_speech_parser'
    data_split = 'turn_dev'  # aka ${PREFIX}

    # append current batch; ensure file does not exist before running!
    with open(f"{proj_home}/{data_split}_embed.txt", "a") as f:

        idx = 0

       	for sent in sentences:

       	    sent_id = sent_ids.pop(0)

            f.write(f"<START> {' '.join([str(x) for x in embed_concat[idx].tolist()])}\n")
            idx += 1

       	    for	i, tok in enumerate(sent):

       	       	f.write(f"{tok[1]}+{sent_id}+{i} {' '.join([str(x) for x in embed_concat[idx].tolist()])}\n")
                idx += 1

       	    f.write(f"<STOP> {' '.join([str(x) for x in embed_concat[idx].tolist()])}\n")
            idx += 1

    # post process with index.py: add vocab length and embed dimension to top of file, remove meta token duplicates

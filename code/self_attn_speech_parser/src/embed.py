"""
Author: Eric DeMattos
Stores all custom embeddings (token + speech features) for each data split.
"""
import torch


def save(token_feats, speech_feats, sentences, sent_ids):
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
            doc_id, _, turn_id = sent_id.split('_')
            sent_id = f"{doc_id}_{turn_id}"

            f.write(f"<START> {' '.join([str(x) for x in embed_concat[idx].tolist()])}\n")
            idx += 1

            for i, tok in enumerate(sent):
                f.write(f"{tok[1]}+{sent_id}+{i} {' '.join([str(x) for x in embed_concat[idx].tolist()])}\n")
                idx += 1

            f.write(f"<STOP> {' '.join([str(x) for x in embed_concat[idx].tolist()])}\n")
            idx += 1

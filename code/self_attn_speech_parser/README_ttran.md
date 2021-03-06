## Paper
This is code used in our Interspeech paper: [On the Role of Style in Parsing Speech with Neural Models](https://ttmt001.github.io/pdfs/3122_Paper.pdf).

## Usage
* `example_job_small.sh` contains an example of how to run the code (training and evaluation). 
* `example_jobs.sh` contains an example of training with multiple seeds (for robustness checks -- see our paper)

Note that these scripts assume the parse trees and speech features are already available. Examples of feature formats are in `sample_data`:
* Parse trees should follow the Penn Treebank format
* Acoustic-prosodic features need to be extracted as described in the Parsing Speech paper. We used Kaldi. Although these can't be used out-of-the box, example codes for the feature extraction pipeline is in https://github.com/trangham283/prosody_nlp/tree/master/code/kaldi_scripts and https://github.com/trangham283/prosody_nlp/tree/master/code/feature_extraction

## Acknowledgements
The code in this repository is based on the implementations in these papers:
1. "Constituency Parsing with a Self-Attentive Encoder", Kitaev and Klein, ACL 2018: 
https://github.com/nikitakit/self-attentive-parser 
2. "Parsing Speech: A Neural Approach to Integrating Lexical and Acoustic-Prosodic Information", Tran et al., NAACL 2018: 
https://github.com/shtoshni92/speech_parsing


## Modifications from original code cited above
* Working with Pytorch 0.4.x instead of 0.3.x or tensorflow 
* Incorporating speech features (CNN module) 


## Other notes:


* The sample data doesn't run currently, since the pause vocab in the train set doesn't include '3' (just by coincidence), but the dev set does. I fixed this by just replacing one of the pause features in the train set with a '3' -- makes the sample data bad, but it makes it possible to run the code and see if everything works.
* In order to run the train script, you have to have the evalb stuff installed. To do this, you should copy the EVALB subdir from https://github.com/nikitakit/self-attentive-parser to https://github.com/trangham283/prosody_nlp/tree/master/code/self_attn_speech_parser/. To install evalb, cd into the EVALB dir and run the command `make`.
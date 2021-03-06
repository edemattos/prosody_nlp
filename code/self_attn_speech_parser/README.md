# Running the parser

## Environment

### Python packages:

1. Create conda environment with python3.6 (or virtual environment) and activate it.
2. Install Pytorch 0.4.1 (command from https://pytorch.org/get-started/previous-versions/):
	
	`conda install pytorch=0.4.1 cuda80 -c pytorch`

3. Install requirements:

`pip install -r requirements.txt`

or

`conda install --file requirements.txt`

### Kaldi: for feature extraction

1. Clone the Kaldi repo: `git clone https://github.com/kaldi-asr/kaldi`
2. Install by following the instructions in the `INSTALL` file.

### PYEVALB: for parser evaluation

1. Clone the (fork of the original) repo: `git clone https://github.com/ekayen/PYEVALB`
2. `cd PYEVALB`
3. `pip install -e .`

## Preprocessing

### Feature extraction

#### Text features:

Generate PTB-style trees with nested parentheses:

1. `cd prosody_nlp/code/feature_extraction`
2. Change file paths in `nxt_proc.py` to point to correct data and output locations.
3. `python2 nxt_proc.py` for each split (train,dev,test). *NOTE: this script requires python2*
4. Change file paths in `make_alignment_dicts.py` to correct data and output locations. This includes the data directory `swb_ms98_transcriptions`, which is part of the original switchboard1 release and not included with the Switchboard NXT annotations. *NOTE: if you encounter errors with this script, try running with python2*

Generate corresponding sentence id files:

4. Change file paths in `make_sent_ids.py`
5. `python make_sent_ids.py`

Download GloVe vectors: 

6. Download glove.6B.300d.txt from https://nlp.stanford.edu/projects/glove/


#### Pitch and intensity

1. `cd prosody_nlp/code/kaldi_scripts`
2. Set the paths in `comp_all.sh` and `paths.sh` to point to the Switchboard data, the kaldi installation, and the directory where you want to output the data.
3. Run `./comp_all.sh`
4. `cd prosody_nlp/code/feature_extraction`
5. Set the variable `nsplit` in `process_kaldi_feats_splits.py` to the number of splits used to generate your kaldi features.
6. Run `process_kaldi_feats_splits.py`. If you had (for example) saved the kaldi features to `raw_output` and want to save the processed kaldi features to `output`, you would use the command: `python process_kaldi_feats_splits.py --in_dir raw_output --out_dir output`.

#### Duration:

1. Create a JSON file of the average duration of each token type in the train set. Since this is based on the SWBD-NXT data, we don't include it. In order to replicate it, create a JSON of the following format using the train set (numbers are dummy values):

```
{
  "okay": {
      "count": 40,
      "mean": 0.7,	
      "std": 0.1
       },
  "uh": {
      "count": 10,
      "mean": 0.5,
      "std": 0.2,
       },
       ...
}						
```
2. `cd prosody_nlp/code/feature_extraction`
3. `python get_ta_stats.py --input_dir <output of make_alignment_dicts.py> --output_dir <desired output dir>`
4. Change file paths in `extract_ta_features.py` to point to extracted kaldi feats. *NOTE: if you encounter errors with this script, try running with python2*
5. `python extract_ta_features.py`


#### Pause

1. cd `prosody_nlp/code/self_attn_speech_parser/src`
2. `python calculate_pauses.py`


### Feature preparation

Put all the features you have generated into a form the parser can use. Even if you are planning on only using turn-based features, you will need to generate both sets -- the turn-based features draw on the sentence-based features.

1. To generate sentence-level features, run `python prep_input_dicts.py` for each split (train,dev,test). Be sure to change file paths to data.
2. To generate turn-level features, run `python prep_turn_dicts.py` for each split (train,dev,test). Be sure to change file paths to data.

Next, filter out turns over 270 tokens (see paper). This should be two turns in the train set, none in other sets.

3. `cd prosody_nlp/code/self_attn_speech_parser/src/`
4. Change data and output paths in `filter_long_turns.py` and make sure it is set to the train split.
5. `python filter_long_turns.py`
6. Make sure that the filtered train sent_id and trees files are the files that you use in the subsequent steps.


## Training

Train script example:

```
python src/main_sparser.py train --use-glove-pretrained --freeze \
       			   	--train-path ${DATA_DIR}/train.trees\
			   	--train-sent-id-path ${DATA_DIR}/train_sent_ids.txt \
				--dev-path ${DATA_DIR}/dev.trees \
				--dev-sent-id-path ${DATA_DIR}/dev_sent_ids.txt \
			        --prefix ${PREFIX} \ 
				--feature-path ${FEAT_DIR} \
				--model-path-base ${MODEL_DIR}/${MODEL_NAME} \
				--speech-features duration,pause,partition,pitch,fbank \
				--sentence-max-len 270 \
				--d-model 1536 \
				--d-kv 96 \
				--morpho-emb-dropout 0.3 \
				--num-layers 4 \
				--num-heads 8 \
				--epochs 50 \
				--numpy-seed $SEED  >> ${RESULT_DIR}/${MODEL_NAME}.log
```

Note: `main_sparser.py` will look for a given speech feature (e.g., pause) at the path `${FEAT_DIR}/train_pause.pickle`; if you want to create subsections of the data, you can use the $PREFIX valiable to point to data at `${FEAT_DIR}/${PREFIX}train_pause.pickle`. Set this variable to '' otherwise. 

## Predicting

```
SENT_ID_PATH=${DATA_DIR}/dev_sent_ids.txt
TREE_PATH=${DATA_DIR}/dev.trees
python src/main_sparser.py test \
    --test-path ${TREE_PATH} \
    --test-sent-id-path ${SENT_ID_PATH} \
    --output-path ${OUTPUT_PATH} \
    --feature-path ${FEAT_DIR} \
    --test-prefix ${PREFIX}_dev \
    --model-path-base ${MODEL_PATH} 

```

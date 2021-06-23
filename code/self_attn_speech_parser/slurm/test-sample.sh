#!/bin/bash
# Author(s): James Owers (james.f.owers@gmail.com)


# Options for sbatch
#SBATCH --gres=gpu:1


# Logging information
# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"


echo "Setting up bash enviroment"
source ~/.bashrc

# Set paths
DATA_DIR=/disk/scratch/s2057915/prosody_nlp/code/self_attn_speech_parser/sample_data/rewrite
MODEL_DIR=/home/s2057915/prosody_nlp/code/self_attn_speech_parser/models
SEED=1234
SENT_ID_PATH=${DATA_DIR}/sample_dev_sent_ids.txt
TREE_PATH=${DATA_DIR}/sample_dev.txt
FEAT_DIR=sample_data
PREFIX=sample_dev
MODEL_PATH=${MODEL_DIR}/turn_sp_correct_eval_72240_dev=90.90.pt

conda activate prosody
echo "Activated conda env: ${CONDA_DEFAULT_ENV}"

# Make script bail out after first error
set -e


# Move from DFS to scratch
SCRATCH_DISK=/disk/scratch
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
mkdir -p ${SCRATCH_HOME}
echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"
# input data directory path on the DFS
proj_home=/home/${USER}/prosody_nlp/code/self_attn_speech_parser
#src_path=${proj_home}/models
# input data directory path on the scratch disk of the node
#dest_path=${SCRATCH_HOME}/prosody_nlp/code/self_attn_speech_parser/models
#mkdir -p ${dest_path}
#rsync --archive --update --compress --progress ${src_path}/ ${dest_path}


# Run experiment
# Read line number ${SLURM_ARRAY_TASK_ID} from the experiment file and run it
# ${SLURM_ARRAY_TASK_ID} is simply the number of the job within the array. If
# you execute `sbatch --array=1:100 ...` the jobs will get numbers 1 to 100
# inclusive.
experiment_text_file=$1
COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" ${experiment_text_file}`"
echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"


# Move from scratch to DFS
echo "Moving output data back to DFS"
# predictions
src_path=${SCRATCH_HOME}/prosody_nlp/code/self_attn_speech_parser/predictions.txt
dest_path=${proj_home}
rsync --archive --update --compress --progress ${src_path} ${dest_path}
# results
src_path=${SCRATCH_HOME}/prosody_nlp/code/self_attn_speech_parser/results
dest_path=${proj_home}
rsync --archive --update --compress --progress ${src_path}/ ${dest_path}

echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"


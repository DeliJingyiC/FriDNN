#!/bin/bash

#SBATCH --time=150:00:00
#SBATCH --account=PAS1957
#SBATCH --gpus-per-node=1
#SBATCH --output=sbatch/%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=28
echo $SLURM_JOB_ID


module load miniconda3/4.10.3-py37
module load cuda/11.2.2

source activate
conda activate tsf114

conda env list

export TF_CPP_MIN_LOG_LEVEL=1

set -x
mkdir output/$SLURM_JOB_ID
zip -r output/$SLURM_JOB_ID/code.zip *.py *.sh train_test

#python train_wavegan.py --model-size 64 --phase-shuffle-shift-factor 2 --post-proc-filt-len 512 --lrelu-alpha 0.2 --valid-ratio 0.1 --test-ratio 0.1 --batch-size 64 --num-epochs 3000 --batches-per-epoch 100 --ngpus 1 --latent-dim 100 --epochs-per-sample 1 --sample-size 20 --learning-rate 1e-4 --beta-one 0.5 --beta-two 0.9 --regularization-factor 10.0 --audio_dir=setTwo --output_dir=output --discriminator-updates=5 --job_id=$SLURM_JOB_ID > "sbatch/${SLURM_JOB_ID}_main.log"
python train_test/main.py --timit_directory=timit --model_dir=model --output_dir=output --job_id=$SLURM_JOB_ID #> "sbatch/${SLURM_JOB_ID}_main.log"
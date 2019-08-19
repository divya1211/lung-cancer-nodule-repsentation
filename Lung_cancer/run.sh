#!/bin/sh

#SBATCH --job-name=lung-nodule
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500GB
#SBATCH --time=10:0:00
#SBATCH --output=./slurm-output/lung-nodule-slurm.%j.out
##SBATCH --gres=gpu:1
##SBATCH --gres=gpu:2080ti:4
##SBATCH --constraint=gpu_12gb
##SBATCH --nodelist=[loopy1, loopy2, loopy3, loopy4, loopy5, loop6, loopy7, loopy8]

#Uncomment to execute python code
#module purge

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv 1>/dev/null 2>&1; then
  eval "$(pyenv init -)"
fi
alias p3="python3"
alias p="pyenv"
p shell anaconda3-5.3.1


ROOT="/scratch/dj1311/Lung_cancer"
cd $ROOT

FILENAME="$1"

CODE="python -u $FILENAME"
eval $CODE;


#!/bin/bash
#SBATCH --job-name=eicu_cuda_job              # Job name
#SBATCH --output=eicu_cuda.out        # Output file
#SBATCH --error=eicu_cuda.err         # Error file
#SBATCH --ntasks=1                               # Number of tasks (processes)
#SBATCH --cpus-per-task=4                        # Number of CPU cores per task
#SBATCH --mem=8G                                 # Total memory limit
#SBATCH --gres=gpu:1                             # Request 1 GPU

# Load necessary modules
# module load python/3.8                          # Load Python module (adjust version as needed)


# Optional: Activate a virtual environment
# Uncomment and modify the following lines if you're using a virtual environment
# source /path/to/your/venv/bin/activate
# Initialize Conda and activate the environment
source /local/scratch/lzeng28/miniconda3/etc/profile.d/conda.sh
conda activate cissl


# Print some environment information (optional but useful for debugging)
echo "Job started on $(date)"
echo "Running on node $(hostname)"
echo "PyTorch CUDA Runtime Version:"
python -c "import torch; print(torch.version.cuda)"


group="test"


python setupdata.py

python transfed.py \
    --lambda_sc 0.0001 \
    --lambda_u 5



# Deactivate Conda environment
conda deactivate

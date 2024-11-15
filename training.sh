#!/usr/bin/env bash
#SBATCH -A naiss2023-22-1358
#SBATCH -p alvis
#SBATCH --gpus-per-node=T4:1  # Request one T4 GPU
#SBATCH -n 1  # Number of tasks
#SBATCH -c 4  # Number of CPU cores per task
#SBATCH -t 8:00:00  # Adjust the time limit as needed
#SBATCH --output=output_kin_nonkin_model/kinship_training_v7_l2.out
#SBATCH --error=output_kin_nonkin_model/kinship_training_v7_l2.err

# Load necessary modules
ml purge  # Ensure we don't have any conflicting modules loaded
ml PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
ml TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
ml cuDNN/8.9.2.26-CUDA-12.1.1
ml JupyterLab/4.0.5-GCCcore-12.3.0

# Create and activate virtual environment if it doesn't exist
if [ ! -d "/cephyr/users/mehdiyev/Alvis/kinship_project/kinship_venv" ]; then
    python -m virtualenv --system-site-packages /cephyr/users/mehdiyev/Alvis/kinship_project/kinship_venv
    source /cephyr/users/mehdiyev/Alvis/kinship_project/kinship_venv/bin/activate
    pip install ipykernel
    python -m ipykernel install --user --name=kinship --display-name="Kinship Project"
else
    source /cephyr/users/mehdiyev/Alvis/kinship_project/kinship_venv/bin/activate
fi

# Run your Python script
cd /cephyr/users/mehdiyev/Alvis/kinship_project/notebooks/
python /cephyr/users/mehdiyev/Alvis/kinship_project/src/kin_nonkin_training_v7.py

#!/usr/bin/env bash
#SBATCH -A naiss2023-22-1358
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1 
#SBATCH -n 1  # Number of tasks
#SBATCH -c 4  # Number of CPU cores per task
#SBATCH -t 12:00:00  # Adjust the time limit as needed
#SBATCH --output=output_kin_nonkin_model2/arcface_embeddings.out
#SBATCH --error=output_kin_nonkin_model2/arcface_embeddings.err

# Load necessary modules
ml purge  # Ensure we don't have any conflicting modules loaded
ml Python/3.12.3-GCCcore-13.3.0
ml virtualenv/20.26.2-GCCcore-13.3.0

# Create and activate virtual environment if it doesn't exist
if [ ! -d "/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/kinship_venv_tf" ]; then
    python -m virtualenv --system-site-packages /cephyr/users/mehdiyev/Alvis/kinship_project/kinship_venv
    source /cephyr/users/mehdiyev/Alvis/kinship_project/kinship_venv/bin/activate
    pip install ipykernel
    python -m ipykernel install --user --name=kinship --display-name="Kinship Project"
else
    source /mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/kinship_venv_tf/bin/activate
fi

# Run Python script
cd /mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/src
python /mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/src/deepface_embeddings_presave.py

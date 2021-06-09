#!/bin/tcsh
#SBATCH --job-name=mg_full_run
#SBATCH --partition=pGPU
#SBATCH --ntasks=4
#SBATCH --mail-user=elisabeth.schiessler@hereon.de
#SBATCH --mail-type=ALL
#SBATCH --output=job.o%j
#SBATCH --error=job.e%j

module load /project/opt/modules/modulefiles/applications/python/3.8
python3 full_automated.py ze41_mol_desc_db_red.csv
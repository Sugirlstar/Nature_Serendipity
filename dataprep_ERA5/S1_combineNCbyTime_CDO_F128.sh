#!/bin/bash
#SBATCH -A ccrc
#SBATCH --partition=cpu
#SBATCH --job-name=mergetime_1dg_F128
#SBATCH --output=out_%j.out
#SBATCH --error=err_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00

module load gcc/11.1.0  
module load openmpi/4.1.6
module load cdo

cdo -b F32 invertlat -mergetime /scratch/bell/hu1029/Data/raw/ERA5_Z500_F128/*.nc /scratch/bell/hu1029/Data/processed/ERA5_Z500_6hr_1979_2021_F128.nc
cdo sinfo /scratch/bell/hu1029/Data/processed/ERA5_Z500_6hr_1979_2021_F128.nc

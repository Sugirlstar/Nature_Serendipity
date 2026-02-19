#!/bin/bash
#SBATCH -A ccrc
#SBATCH --partition=cpu
#SBATCH --job-name=mergetime_1dg_F128
#SBATCH --output=out_%j.out
#SBATCH --error=err_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=5:00:00

module load gcc/11.1.0  
module load openmpi/4.1.6
module load cdo

RAW_DIR=/scratch/bell/hu1029/Data/raw/MERRA2/Z500
OUT_DIR=/scratch/bell/hu1029/Data/processed
OUT_temp=/scratch/bell/hu1029/Data/processed/MERRA2_temp
mkdir -p $OUT_temp

# 01: merge all the files into one file and regrid
cdo -b F32 mergetime $RAW_DIR/MERRA2_*.nc $OUT_temp/MERRA2_Z500.nc
# select only 1980-2021
cdo seldate,1980-01-01,2021-12-31 $OUT_temp/MERRA2_Z500.nc $OUT_temp/MERRA2_Z500_6hourly_1980_2021.nc
# regrid to 1dg and F128
cdo -b F32 remapbil,r360x181 $OUT_temp/MERRA2_Z500_6hourly_1980_2021.nc $OUT_DIR/MERRA2_Z500_6hr_1980_2021_1dg.nc
cdo -b F32 invertlat -remapbil,n128 $OUT_temp/MERRA2_Z500_6hourly_1980_2021.nc $OUT_DIR/MERRA2_Z500_6hr_1980_2021_F128.nc

rm -rf $OUT_temp


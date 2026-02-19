#!/bin/bash
#SBATCH -A ccrc
#SBATCH --partition=cpu
#SBATCH --job-name=makesingleyearF128
#SBATCH --output=out_%j.out
#SBATCH --error=err_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=3:00:00

module load gcc/11.1.0  
module load openmpi/4.1.6
module load cdo

RAW_DIR=/scratch/bell/hu1029/Data/raw/MERRA2/Z500
OUT_DIR=/scratch/bell/hu1029/LGHW/TRACK/MERRA2_TRACK_inputdata_geopotential_yearly
mkdir -p "$OUT_DIR"

for y in $(seq 1980 2021); do
  echo "Processing $y ..."

  files=$(ls "$RAW_DIR"/MERRA2_*.${y}[0-1][0-9][0-3][0-9]_Z500.nc 2>/dev/null | sort)
  [ -z "$files" ] && { echo "[WARN] no files for $y"; continue; }

  cdo -b F32 remapbil,n128 -mulc,9.80665 -mergetime \
    $files \
    "$OUT_DIR/MERRA2_Z500_6hr_${y}_F128.nc"
done

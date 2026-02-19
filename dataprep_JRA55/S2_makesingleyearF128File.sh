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

RAW_DIR=/scratch/bell/hu1029/Data/raw/JRA55/Z
OUT_DIR=/scratch/bell/hu1029/LGHW/TRACK/JRA55_TRACK_inputdata_geopotential_yearly
TMP_DIR=/scratch/bell/hu1029/tmp_jra55

mkdir -p "$OUT_DIR"

for dir in "$RAW_DIR"/*/; do
  y=$(basename "$dir")
  echo "Processing $y"

  rm -rf "$TMP_DIR"
  mkdir -p "$TMP_DIR"

  # take level 50000 and regrid to F128, multiply by 9.81 to convert from geopotential to geopotential height
  i=0
  for f in "$dir"/anl_p125.007_hgt*; do
    [ -f "$f" ] || continue
    i=$((i+1))
    cdo -O -b F32 -remapbil,n128 -sellevel,50000 -mulc,9.80665 "$f" "$TMP_DIR/${i}.nc" 
  done

  # merge
  cdo -O -f nc mergetime "$TMP_DIR"/*.nc "$OUT_DIR/JRA55_Z500_6hr_${y}_F128.nc"

done

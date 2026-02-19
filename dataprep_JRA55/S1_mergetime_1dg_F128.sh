#!/bin/bash
#SBATCH -A ccrc
#SBATCH --partition=cpu
#SBATCH --job-name=mergetime_1dg_F128_jra55
#SBATCH --output=out_%j.out
#SBATCH --error=err_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=6:00:00

module load gcc/11.1.0  
module load openmpi/4.1.6
module load cdo

RAW_DIR=/scratch/bell/hu1029/Data/raw/JRA55/Z
OUT_DIR=/scratch/bell/hu1029/Data/processed
OUT_temp=/scratch/bell/hu1029/Data/processed/tempjra55
mkdir -p $OUT_temp

# 01: regrid to 1dg and F128, and sellevel,50000
# temp subfolders
TMP_1DG="$OUT_temp/tmp_1dg"
TMP_F128="$OUT_temp/tmp_f128"
mkdir -p "$TMP_1DG" "$TMP_F128"

# loop over raw files (recursive)
find "$RAW_DIR" -type f -name "anl_p125.007_hgt*" | sort | while read -r f; do

  base=$(basename "$f")

  # 1) sellevel + regrid to 1deg (process file saved in OUT_temp)
  cdo -b F32 remapbil,r360x181 -sellevel,50000 \
    "$f" \
    "$TMP_1DG/${base}_z500_1dg.nc"

  # 2) sellevel + regrid to F128 (process file saved in OUT_temp)
  cdo -b F32 invertlat -remapbil,n128 -sellevel,50000 \
    "$f" \
    "$TMP_F128/${base}_z500_f128.nc"
done

# merge (already regridded + 500hPa only)
cdo -f nc -b F32 mergetime "$TMP_1DG"/*_z500_1dg.nc "$OUT_temp/JRA55_Z500_6hr_1dg_all.nc"
cdo -f nc -b F32 mergetime "$TMP_F128"/*_z500_f128.nc "$OUT_temp/JRA55_Z500_6hr_f128_all.nc"

# select only 1979-2021 and write final outputs
cdo -f nc seldate,1979-01-01,2021-12-31 \
  "$OUT_temp/JRA55_Z500_6hr_1dg_all.nc" \
  "$OUT_DIR/JRA55_Z500_6hr_1979_2021_1dg.nc"

cdo -f nc seldate,1979-01-01,2021-12-31 \
  "$OUT_temp/JRA55_Z500_6hr_f128_all.nc" \
  "$OUT_DIR/JRA55_Z500_6hr_1979_2021_F128.nc"

rm -rf $OUT_temp


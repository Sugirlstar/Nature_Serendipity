#!/bin/bash
#SBATCH -A ccrc
#SBATCH --partition=cpu
#SBATCH --job-name=combine_NC_by5years_CDO_F128
#SBATCH --output=out_%j.out
#SBATCH --error=err_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00

module load gcc/11.1.0  
module load openmpi/4.1.6
module load cdo

RAW_DIR=/scratch/bell/hu1029/Data/raw/MERRA2/Z500
TRACK_OUT_DIR=/scratch/bell/hu1029/LGHW/TRACK/MERRA2_TRACK_inputdata_geopotential
OUT_temp=/scratch/bell/hu1029/Data/processed/temp
mkdir -p $OUT_temp
mkdir -p $TRACK_OUT_DIR

# 02: merge files by 5 years for tracking input data
END_YEAR=2021

for start in $(seq 1980 5 $END_YEAR); do
  end=$((start + 4))
  if [ $end -gt $END_YEAR ]; then end=$END_YEAR; fi

  echo "Merging $start–$end"

  cdo mergetime \
    ${RAW_DIR}/MERRA2*${start}????.nc \
    $(for y in $(seq $((start+1)) $end); do echo ${RAW_DIR}/MERRA2*${y}????.nc; done) \
    ${OUT_temp}/MERRA2_z500_${start}_${end}.nc

    cdo -b F32 remapbil,n128 ${OUT_temp}/MERRA2_z500_${start}_${end}.nc ${TRACK_OUT_DIR}/MERRA2_Z500_F128_${start}_${end}.nc
done

rm -rf $OUT_temp
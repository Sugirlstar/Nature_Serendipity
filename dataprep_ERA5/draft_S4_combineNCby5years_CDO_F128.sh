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

RAW_DIR=/scratch/bell/hu1029/Data/raw/ERA5_Z500_F128
OUT_DIR=/scratch/bell/hu1029/LGHW/TRACK/TRACK_inputdata_geopotential
END_YEAR=2021

for start in $(seq 1979 5 $END_YEAR); do
  end=$((start + 4))
  if [ $end -gt $END_YEAR ]; then end=$END_YEAR; fi

  files=()
  for y in $(seq $start $end); do
    files+=("${RAW_DIR}/ERA5_Z500_6hr_${y}.nc")
  done

  cdo mergetime "${files[@]}" \
    "${OUT_DIR}/ERA5_Z500_6hr_${start}_${end}_F128.nc"
done

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


# 1. divide into single-year files
# 2. flip latitudes from -90~90 to 90~-90
# 3. make it geopotential (units: m^2/s^2) instead of geopotential height (units: m)

INPUT_File=/scratch/bell/hu1029/Data/processed/MERRA2_Z500anomaly_subtractseasonal_6hr_1980_2021_F128.nc
OUTPUT_DIR=/scratch/bell/hu1029/LGHW/TRACK/MERRA2_TRACK_inputdata_geopotentialAnomaly_yearly


echo "Step 2: split yearly"

for year in $(seq 1980 2021); do
    echo "Processing $year"

    OUTFILE=${OUTPUT_DIR}/MERRA2_geopotentialAnomaly_6hr_${year}.nc

    cdo -O invertlat -mulc,9.80665 -selyear,$year "$INPUT_File" "$OUTFILE"
done

rm -f "$TMP_ALL"

echo "[DONE]"

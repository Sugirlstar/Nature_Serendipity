#!/bin/bash

module load gcc/11.1.0  
module load openmpi/4.1.6
module load cdo

# Regrid Z500 data to 1 degree grid
cdo remapbil,r360x181 /scratch/bell/hu1029/Data/processed/ERA5_Z500_6hr_1979_2021_regulargrid_Float32.nc /scratch/bell/hu1029/Data/processed/ERA5_Z500_6hr_1979_2021_1dg.nc
cdo sinfo /scratch/bell/hu1029/Data/processed/ERA5_Z500_6hr_1979_2021_1dg.nc

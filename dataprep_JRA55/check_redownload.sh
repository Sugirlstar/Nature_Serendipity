#!/bin/bash
#SBATCH -A ccrc
#SBATCH --partition=cpu
#SBATCH --job-name=rebuild_badfiles
#SBATCH --output=out_%j.out
#SBATCH --error=err_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

BASE_URL="https://osdf-director.osg-htc.org/ncar/gdex/d628000/anl_p125"
export BASE_URL

module load parallel

cat bad_files.log | parallel -j 10 '
filepath={}
filename=$(basename "$filepath")
rm -f "$filepath"
year=$(echo "$filename" | cut -d'.' -f3 | cut -c1-4)
url="$BASE_URL/$year/$filename"
echo "Downloading $filename"
wget $cert_opt $opts "$url"
mv "$filename" "$filepath"
' 

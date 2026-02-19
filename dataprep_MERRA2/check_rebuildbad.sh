#!/bin/bash
set -euo pipefail

module load gcc/11.1.0  
module load openmpi/4.1.6
module load cdo

BADLIST="bad_files.log"

RAW_Z500_DIR="/scratch/bell/hu1029/Data/raw/MERRA2/Z500"
RAW_ALL_DIR="/scratch/bell/hu1029/Data/raw/MERRA2/ALLVars"

while IFS= read -r OUTFILE; do

  [[ -z "$OUTFILE" ]] && continue

  base=$(basename "$OUTFILE")                 # e.g. MERRA2_...19880125_Z500.nc
    inbase="${base/_Z500.nc/.nc4}"           # replace with .nc4
    INFILE="$RAW_ALL_DIR/$inbase"

  if [[ ! -f "$INFILE" ]]; then
    echo "MISSING INFILE: $INFILE (for OUTFILE: $OUTFILE)"
    continue
  fi

  echo "Fixing:"
  echo "  IN : $INFILE"
  echo "  OUT: $OUTFILE"

  cdo -b F32 -sellevel,500 -selname,H "$INFILE" "$OUTFILE"

done < "$BADLIST"

echo "Done."
#!/bin/bash
#SBATCH -A ccrc
#SBATCH --partition=cpu
#SBATCH --job-name=download
#SBATCH --output=out_%j.out
#SBATCH --error=err_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00

module load gcc/11.1.0  
module load openmpi/4.1.6
module load cdo
module load parallel

OUT_DIR=/scratch/bell/hu1029/Data/raw/MERRA2/Z500
TEMP_DIR=/scratch/bell/hu1029/Data/raw/MERRA2/ALLVars
URL_LIST=subset_M2I6NPANA_5.12.4_20260211_192421_.txt
mkdir -p $TEMP_DIR
mkdir -p $OUT_DIR

# Convert Windows line endings to Unix line endings
dos2unix "$URL_LIST"


JOBS=${SLURM_CPUS_PER_TASK:-5} # in slurm job, use all allocated CPUs; otherwise, default to 5
echo "JOBS=$JOBS"

export TEMP_DIR OUT_DIR

cat "$URL_LIST" | parallel -j "$JOBS" --linebuffer '
  URL={}
  FNAME=$(basename "$URL")
  INFILE="$TEMP_DIR/$FNAME"
  BASE=${FNAME%.nc4}
  OUTFILE="$OUT_DIR/${BASE}_Z500.nc"

  echo "[START] $FNAME"

  # 1) download file if not exists or is empty
  if [ ! -s "$INFILE" ]; then
    wget --load-cookies "$HOME/.urs_cookies" \
        --save-cookies  "$HOME/.urs_cookies" \
        --keep-session-cookies \
        --auth-no-challenge=on \
        --content-disposition \
        -P "$TEMP_DIR" \
        "$URL" || exit 1
    echo "[DL] $FNAME downloaded"

    # change file name
    RAW=$(basename "$URL")
    CLEAN=${RAW%%\?*}
    DOWNLOADED=$(find "$TEMP_DIR" -maxdepth 1 -type f -name "${RAW}*" | head -n 1)
    if [ -n "$DOWNLOADED" ]; then
        mv "$DOWNLOADED" "$TEMP_DIR/$CLEAN"
    fi

    # 2) process: select variable H + select 500 hPa
    cdo -b F32 -sellevel,500 -selname,H "$INFILE" "$OUTFILE" || exit 2
    echo "[DONE] $TEMP_DIR/$CLEAN -> $OUTFILE"

  else
    echo "[SKIP DL] $FNAME exists"
  fi

'

echo "###########ALL DONE###########"
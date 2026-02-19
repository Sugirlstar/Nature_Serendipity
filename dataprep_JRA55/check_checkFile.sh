#!/bin/bash
#SBATCH -A ccrc
#SBATCH --partition=cpu
#SBATCH --job-name=check_nc_parallel
#SBATCH --output=out_%j.out
#SBATCH --error=err_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=8G
#SBATCH --time=00:30:00

module load gcc/11.1.0
module load openmpi/4.1.6
module load cdo
module load parallel

DIR="/scratch/bell/hu1029/Data/raw/JRA55/Z"
FILELIST="filelist.txt"
BADLIST="bad_files.log"
ERRLOG="cdo_errors.log"

# make file list
find "$DIR" -type f -name "anl*" | sort > "$FILELIST"

# clear bad list and error log
: > "$BADLIST"
: > "$ERRLOG"

# parallel check: if error, print filename to BADLIST, and append cdo stderr to ERRLOG
parallel -j "$SLURM_CPUS_PER_TASK" --line-buffer '
  f={}
  if ! cdo sinfo "$f" >/dev/null 2>>"'"$ERRLOG"'"; then
    echo "$f" >>"'"$BADLIST"'"
  fi
' :::: "$FILELIST"

echo "Done."
echo "Bad files list: $BADLIST"
echo "Errors (optional): $ERRLOG"
echo "Count bad: $(wc -l < "$BADLIST")"

#!/bin/bash
#SBATCH -A ccrc
#SBATCH --partition=cpu
#SBATCH --job-name=multiprocess_TRACK
#SBATCH --output=multiprocess_%j.out
#SBATCH --error=multiprocess__%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00

module load parallel

cd /scratch/bell/hu1029/LGHW/TRACK/track-TRACK-1.5.4
export PATH="${PATH}:."
chmod +x ./master bin/track.linux
chmod +x ./config.RUN

DATA="/scratch/bell/hu1029/LGHW/TRACK/TRACK_inputdata_geopotentialAnomaly"
TRACKS="${DATA}/TRACKS_SH"
if [ ! -d "$TRACKS" ]; then
  mkdir "$TRACKS"
fi

# get all the nc files (only the file name)
files=($(find "$DATA" -type f -name "*.nc" -exec basename {} \;))

for ZFILE in "${files[@]}"; do
  if [ ! -e indat/$ZFILE ]; then
    ln -s $DATA/$ZFILE indat/$ZFILE
  else
    echo "Symbolic link for $ZFILE already exists"
  fi
done

echo ${files[@]} | tr ' ' '\n' | parallel -j 128 --env TRACKS "
  ZFILE={};
  TRACKS=/scratch/bell/hu1029/LGHW/TRACK/TRACK_inputdata_geopotentialAnomaly/TRACKS_SH;
  STUB=\$(echo \$ZFILE | sed -e 's/\.nc//');
  FILT42=\${STUB}_zfilt_T42.dat;
  EXT=\$STUB;
  bin/track.linux -i \$ZFILE -f \$EXT < specfilt_vor.in;
  mv outdat/specfil.\${EXT}_band001 indat/\$FILT42;
  master -c=\${STUB}_zonefilt_T42_SH -e=track.linux -d=now -i=\$FILT42 -f=\$EXT -j=RUN_AT.in -k=initial.T42_SH -n=1,62,24 -o=\$TRACKS -r=RUN_AT_ -s=RUNDATIN.6hr_Z_T42
"

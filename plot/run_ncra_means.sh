#!/bin/bash
# Precompute the full-record time-mean w/NO3 fields (from the zsliced
# product) that plot_cs_wno3_aktdno3dz_snap.py's load_ncra_mean() looks for,
# using NCO's ncra instead of that script's slow Python full-record scan.
# Run this from the plot/ directory: bash run_ncra_means.sh
#
# Saved as a file (not pasted into an interactive shell) specifically to
# avoid multi-line-paste corruption of the declare -A / $(...) syntax below.

set -e

ZSLICE_ROOT=/data/project1/minnaho/swel/zslicefull
OUT_DIR=figs/cs_wno3_aktdno3dz_snap/ncra_means
mkdir -p "$OUT_DIR"

declare -A ZDIR=( [tideswec]=tideswec [tidesnowec]=tidesnowec \
                  [notidesnowec]=notidesnowec [notideswec]=notideswec \
                  [ampwec]=notidesampwec [tidesampwec]=tidesampwec )

for scen in tideswec tidesnowec notidesnowec notideswec ampwec tidesampwec; do
    zdir=${ZDIR[$scen]}
    his=$(ls "$ZSLICE_ROOT/$zdir"/z_mc60_his.*.nc | grep -v 20190429110056)
    bgc=$(ls "$ZSLICE_ROOT/$zdir"/bgc/z_mc60_bgc.*.nc | grep -v dia_avg | grep -v 20190429110056)
    echo "[$scen] w mean: $(echo "$his" | wc -l) files"
    echo "[$scen] NO3 mean: $(echo "$bgc" | wc -l) files"
    ncra -O -v w,depth   $his "$OUT_DIR/w_mean_${scen}.nc"   &
    ncra -O -v NO3,depth $bgc "$OUT_DIR/no3_mean_${scen}.nc" &
done
wait
echo "done -- see $OUT_DIR"

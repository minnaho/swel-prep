#!/bin/bash
# Relaunch all plot_cs_diag*.py / plot_cs_ini.py / plot_cs_wno3_aktdno3dz_snap.py
# scripts simultaneously, now that the domain-averaged depth-layer filter
# (depth_mean/keep) has been stripped from each so the plotted bathymetry
# matches the actual seafloor instead of a mean-across-transect artifact.
#
# None of these are registered in run_plots.py, so they're launched directly
# here. Each runs as its own background process, logging to logs/<script>.log;
# this script exits once all of them finish.
#
# Run inside a screen session so it survives a disconnect:
#   cd /data/project3/minnaho/project9copy/swel/plot
#   screen -dmS cs_replots bash run_cs_replots.sh
#   screen -r cs_replots      # reattach to check progress
#   tail -f logs/*.log        # or just watch the logs directly

cd "$(dirname "$0")"
mkdir -p logs

SCRIPTS=(
    plot_cs_diag.py
    plot_cs_diag_no3.py
    plot_cs_diag_o2.py
    plot_cs_diag_totc.py
    plot_cs_diag_rho.py
    plot_cs_diag_akt.py
    plot_cs_diag_akv.py
    plot_cs_diag_bgcdia.py
    plot_cs_ini.py
    plot_cs_wno3_aktdno3dz_snap.py
)

for f in "${SCRIPTS[@]}"; do
    python3 "$f" > "logs/${f%.py}.log" 2>&1 &
done
wait
echo "done -- see logs/"

#!/bin/bash
set -e
cd /data/project3/minnaho/project9copy/swel

# ============================================================
# Wave 1 -- independent scripts, no shared data dependencies.
# Launch all in parallel, then wait for the postprocessing ones
# to finish before Wave 2 (which reads their npz output).
# ============================================================
mkdir -p logs

(cd plot && python plot_cs_vorticity_snap.py            > ../logs/cs_vort_snap.log 2>&1) &
(cd plot && python plot_surf_ptrace.py                  > ../logs/surf_ptrace.log 2>&1) &
(cd plot && python plot_surf_rtrace.py                  > ../logs/surf_rtrace.log 2>&1) &
(cd postprocessing && python offshore_flux.py               > ../logs/offshore_flux.log 2>&1) &
(cd postprocessing && python offshore_flux_ptrace.py         > ../logs/offshore_flux_ptrace.log 2>&1) &
(cd postprocessing && python offshore_flux_rtrace.py         > ../logs/offshore_flux_rtrace.log 2>&1) &
(cd postprocessing && python offshore_flux_ptrace_zslice.py  > ../logs/offshore_flux_ptrace_zslice.log 2>&1) &
(cd postprocessing && python offshore_flux_rtrace_zslice.py  > ../logs/offshore_flux_rtrace_zslice.log 2>&1) &
(cd postprocessing && python offshore_flux_zslice.py         > ../logs/offshore_flux_zslice.log 2>&1) &
(cd postprocessing && python calc_wtrace_flux.py             > ../logs/calc_wtrace_flux.log 2>&1) &
(cd postprocessing && python calc_wno3_flux_10m.py            > ../logs/calc_wno3_flux_10m.log 2>&1) &
(cd postprocessing && python calc_wno3_flux_20m.py            > ../logs/calc_wno3_flux_20m.log 2>&1) &

wait
echo "Wave 1 done."

# ============================================================
# Wave 2 -- plot scripts that read Wave 1's npz output.
# Each has a hardcoded TRACER (or tracer) variable instead of a
# CLI arg, so each script loops its own tracers SEQUENTIALLY
# inside one background job (sed-editing the same file from two
# processes at once would race) -- but the different SCRIPTS
# still run in parallel with each other.
# ============================================================
cd plot

set_tracer() {  # $1 = file, $2 = value -- matches TRACER = '...'
  sed -i "s/^\(TRACER[[:space:]]*=[[:space:]]*\)'[^']*'/\1'$2'/" "$1"
}
set_tracer_lc() {  # lowercase `tracer = '...'` (plot_wtrace_env.py only)
  sed -i "s/^\(tracer[[:space:]]*=[[:space:]]*\)'[^']*'/\1'$2'/" "$1"
}

(for T in ptrace rtrace NO3; do set_tracer plot_offshore_flux_hov.py $T;                 python plot_offshore_flux_hov.py;                 done) > ../logs/hov.log 2>&1 &
(for T in ptrace rtrace NO3; do set_tracer plot_offshore_flux_hov_time.py $T;            python plot_offshore_flux_hov_time.py;            done) > ../logs/hov_time.log 2>&1 &
(for T in ptrace rtrace NO3; do set_tracer plot_offshore_flux_profile.py $T;             python plot_offshore_flux_profile.py;             done) > ../logs/profile.log 2>&1 &
(for T in ptrace rtrace NO3; do set_tracer plot_offshore_flux_hov_zslice_norm.py $T;     python plot_offshore_flux_hov_zslice_norm.py;     done) > ../logs/hov_zslice_norm.log 2>&1 &
(for T in ptrace rtrace NO3; do set_tracer plot_offshore_flux_hov_time_zslice_norm.py $T; python plot_offshore_flux_hov_time_zslice_norm.py; done) > ../logs/hov_time_zslice_norm.log 2>&1 &
(for T in ptrace rtrace NO3; do set_tracer plot_offshore_flux_profile_zslice_norm.py $T; python plot_offshore_flux_profile_zslice_norm.py; done) > ../logs/profile_zslice_norm.log 2>&1 &
(for T in ptrace rtrace;     do set_tracer_lc plot_wtrace_env.py $T;                     python plot_wtrace_env.py;                        done) > ../logs/wtrace_env.log 2>&1 &
(python plot_wno3_env.py  > ../logs/wno3_env.log 2>&1) &
(python plot_wno3_flux.py > ../logs/wno3_flux.log 2>&1) &

wait
echo "Wave 2 done."

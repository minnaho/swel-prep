#!/bin/bash
# Add PAR to the existing zslice_profiles_bgcdia_100m_offshore.py output
# without re-accumulating every other variable -- see the --vars flag added
# to that script. Launches all 3 scenarios in parallel; merges into each
# scenario's existing npz rather than overwriting it.
#
# Run this, wait for all 3 to finish, then re-run:
#   python -u profile_zslice_bgcdia_100m_offshore.py --merge
# to fold PAR into the combined/coastal outputs too.

cd "$(dirname "$0")"

for scen in tidesampwec tidesnowec notidesnowec; do
    python -u profile_zslice_bgcdia_100m_offshore.py "$scen" --vars PAR > "log_bgcdia_PAR_${scen}.txt" 2>&1 &
done
wait
echo "done -- see log_bgcdia_PAR_*.txt"

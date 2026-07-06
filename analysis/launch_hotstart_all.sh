#!/usr/bin/env bash
# Launch the Wright-protocol (hot-start) re-run for all 24 model x case
# combinations in the background. Outputs -> analysis/hotstart_outputs/.
#   protocol per run: 500-step cold start -> RMSE<3 filter/resample ->
#                     50000 production -> discard 1000 burn-in
#
# Prerequisite: each models/<m>/run.py has already been executed, so the per-case
# executed notebooks (<prefix>_executed_<case>.ipynb) exist in each model dir.
# These 24 runs are heavy (50k steps each); launched together they will saturate
# the machine -- reduce the set or run sequentially if that is a problem.
set -u

# Repo root, resolved relative to this script (analysis/ is one level below root).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/analysis/run_hotstart.py"
OUTDIR="$ROOT/analysis/hotstart_outputs"
mkdir -p "$OUTDIR"

# model directory : executed-notebook prefix (from each model's run.py out_tpl)
MODELS=(
  "1_SCM_Voigt:bm"
  "2_SCM_Hill:bmw"
  "3_HS_Voigt:hsu"
  "4_HS_Hill:hsu"
  "5_DEM_Voigt:demni"
  "6_DEM_Hill:demni"
  "7_KT:kt"
  "8_VRH:vrh_mod_5"
)
CASES=( A:A_constraints_away B:B_wright_inherited C:C_insight_marsquake )

for entry in "${MODELS[@]}"; do
  mdir=${entry%%:*}; pre=${entry#*:}
  for cc in "${CASES[@]}"; do
    cl=${cc%%:*}; cf=${cc#*:}
    nb="$ROOT/models/$mdir/${pre}_executed_${cf}.ipynb"
    out="$OUTDIR/${mdir}_${cl}.npz"
    log="$OUTDIR/${mdir}_${cl}.log"
    if [[ -f "$nb" ]]; then
      # cd into the model dir so the notebook's relative paths resolve as under run.py
      ( cd "$ROOT/models/$mdir" && \
        nohup python3 "$RUN" "${pre}_executed_${cf}.ipynb" "$out" 50000 1000 500 3 > "$log" 2>&1 & )
      echo "launched $mdir $cl"
    else
      echo "MISSING notebook: $nb  (run models/$mdir/run.py first)"
    fi
  done
done
echo "all launched -> $OUTDIR"

#!/usr/bin/env bash
# Launch all 48 binary-saturation runs (8 theories x 3 cases x {wet,dry}),
# throttled to 22 concurrent jobs. Each writes to <dir>/outputs_binary_<em>/.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASES=(A_constraints_away B_wright_inherited C_insight_marsquake)
EMS=(wet dry)
# theory dir list
DIRS=(models/1_SCM_Voigt models/2_SCM_Hill models/3_HS_Voigt models/4_HS_Hill models/5_DEM_Voigt models/6_DEM_Hill models/7_KT models/8_VRH)

# pre-create output dirs so log redirects work
for d in "${DIRS[@]}"; do for em in "${EMS[@]}"; do mkdir -p "$ROOT/$d/outputs_binary_$em"; done; done

# build job list
jobs_file=$(mktemp)
for d in "${DIRS[@]}"; do
  for em in "${EMS[@]}"; do
    for c in "${CASES[@]}"; do
      echo "cd '$ROOT/$d' && python3 run_binary_${em}.py ${c} > 'outputs_binary_${em}/run_${c}.log' 2>&1 && echo 'DONE ${d} ${em} ${c}'" >> "$jobs_file"
    done
  done
done

echo "Launching $(wc -l < "$jobs_file") jobs, 22 concurrent..."
cat "$jobs_file" | xargs -P 22 -I {} bash -c '{}'
echo "ALL BINARY RUNS COMPLETE"
rm -f "$jobs_file"

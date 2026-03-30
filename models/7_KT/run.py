#!/usr/bin/env python
# coding: utf-8

# In[1]:


# run_kt_cases.py
import os
import re
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# 1) Path to your KT notebook (update to your exact filename)
orig_path = "kt.ipynb"   # e.g., "Kuster_Toksoz.ipynb"

# 2) Output naming
out_tpl    = "kt_executed_{name}.ipynb"
output_dir = "outputs_kt"

# 3) Three cases (Vp, Vs in km/s; sigmas in m/s)
cases = [
    {"name": "A_constraints_away",
     "Vp_kms": 3.8, "Vs_kms": 2.2,
     "sigma_Vp": 1000.0, "sigma_Vs": 400.0},

    {"name": "B_wright_inherited",
     "Vp_kms": 4.1, "Vs_kms": 2.5,
     "sigma_Vp": 200.0, "sigma_Vs": 300.0},

    {"name": "C_insight_marsquake",
     "Vp_kms": 4.7, "Vs_kms": 2.7,
     "sigma_Vp": 300.0, "sigma_Vs": 100.0},
]

# 4) Regex patterns for BOTH notebook styles
# (a) Monolithic: d = np.array([...]), s = np.array([...])
pat_d = re.compile(r"^\s*d\s*=\s*np\.array\([^)]*\)\s*$", re.MULTILINE)
pat_s = re.compile(r"^\s*s\s*=\s*np\.array\([^)]*\)\s*$", re.MULTILINE)

# (b) Separate variables typical in your KT notebook
pat_vp  = re.compile(r"^\s*Vp_obs\s*=\s*[^#\n]+", re.MULTILINE)
pat_vs  = re.compile(r"^\s*Vs_obs\s*=\s*[^#\n]+", re.MULTILINE)
pat_svp = re.compile(r"^\s*sigma_Vp\s*=\s*[^#\n]+", re.MULTILINE)
pat_svs = re.compile(r"^\s*sigma_Vs\s*=\s*[^#\n]+", re.MULTILINE)
# If you also keep rho in KT, add patterns for rho here.

def build_patched_nb(nb, case_name, d_vals, s_vals, output_dir):
    """
    Create a new notebook:
      - preamble (Agg + savefig monkey-patch + case tag)
      - original cells with replacements for either style
      - tail that saves any open figs AND .npy arrays if present
    """
    patched = nbformat.v4.new_notebook()
    patched.metadata = nb.metadata

    preamble = f"""
# --- injected preamble (headless + tagging) ---
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

case_name  = {case_name!r}
output_dir = {output_dir!r}
os.makedirs(output_dir, exist_ok=True)

# savefig monkey-patch: force output_dir + tag with case_name
_ORIG_SAVEFIG = plt.Figure.savefig
def _savefig(self, fname, *args, **kwargs):
    base = os.path.basename(str(fname))
    root, ext = os.path.splitext(base)
    if case_name not in root:
        root = f"{{root}}_{{case_name}}"
    out = os.path.join(output_dir, root + (ext or ".png"))
    return _ORIG_SAVEFIG(self, out, *args, **kwargs)
plt.Figure.savefig = _savefig

print(f">> Running case '{{case_name}}'; outputs -> {{output_dir}}")
"""
    patched.cells.append(nbformat.v4.new_code_cell(preamble))

    # Replacement lines (monolithic)
    d_line = f"d = np.array([{d_vals[0]:.6g}, {d_vals[1]:.6g}])"  # KT usually Vp, Vs only
    s_line = f"s = np.array([{s_vals[0]:.6g}, {s_vals[1]:.6g}])"

    # Replacement lines (separate variables; convert km/s -> m/s)
    vp_line = f"Vp_obs = {d_vals[0]*1000.0:.6g}"
    vs_line = f"Vs_obs = {d_vals[1]*1000.0:.6g}"
    svp_line = f"sigma_Vp = {s_vals[0]:.6g}"
    svs_line = f"sigma_Vs = {s_vals[1]:.6g}"

    for cell in nb.cells:
        if cell.cell_type != "code":
            patched.cells.append(cell)
            continue

        src = cell.source

        # Try monolithic d/s (harmless if not present)
        new_src = pat_d.sub(d_line, src)
        new_src = pat_s.sub(s_line, new_src)

        # Try separate variables
        new_src = pat_vp.sub(vp_line, new_src)
        new_src = pat_vs.sub(vs_line, new_src)
        new_src = pat_svp.sub(svp_line, new_src)
        new_src = pat_svs.sub(svs_line, new_src)

        patched.cells.append(nbformat.v4.new_code_cell(new_src))

    tail = r"""
# --- tail: save open figs and .npy arrays if present ---
import os, numpy as np, matplotlib.pyplot as plt

# Save any currently open figures
nums = plt.get_fignums()
for i, n in enumerate(nums, 1):
    fig = plt.figure(n)
    out = os.path.join(output_dir, f"{case_name}_fig{i}.png")
    fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close('all')
print(f">> Saved {len(nums)} figures to {output_dir}")

# Helper: save arrays if they exist
def _maybe_save(var_name, out_stem):
    if var_name in globals():
        try:
            arr = np.asarray(globals()[var_name])
            out = os.path.join(output_dir, f"{out_stem}_{case_name}.npy")
            np.save(out, arr)
            print(f">> Saved {os.path.basename(out)} shape={arr.shape}")
        except Exception as e:
            print(f">> Could not save {var_name} -> {out_stem}: {e}")

# Saturation arrays -> saturation_kt_<CASE>.npy
_maybe_save('idx_water',      'saturation_kt')
_maybe_save('idx_saturation', 'saturation_kt')  # fallback name

# Porosity arrays -> porosity_kt_<CASE>.npy
_maybe_save('idx_porosity', 'porosity_kt')
_maybe_save('idx_por',      'porosity_kt')      # fallback name
"""
    patched.cells.append(nbformat.v4.new_code_cell(tail))
    return patched

# 5) Read original KT notebook
nb_orig = nbformat.read(orig_path, as_version=4)


# 6b) Filter cases if a case name is provided via CLI
import sys
if len(sys.argv) > 1:
    case_filter = sys.argv[1]
    cases = [c for c in cases if c["name"] == case_filter]
    if not cases:
        print(f"ERROR: Unknown case '{case_filter}'")
        sys.exit(1)

# 7) Run cases
os.makedirs(output_dir, exist_ok=True)
for c in cases:
    d_vals = (c["Vp_kms"], c["Vs_kms"])
    s_vals = (c["sigma_Vp"], c["sigma_Vs"])

    nb_p = build_patched_nb(nb_orig, c["name"], d_vals, s_vals, output_dir)

    try:
        client = NotebookClient(nb_p, timeout=86400, kernel_name="python3")
        client.execute()
    except CellExecutionError as e:
        print(f"[{c['name']}] Execution error:\n{e}")

    out_nb = out_tpl.format(name=c["name"])
    with open(out_nb, "w") as f:
        nbformat.write(nb_p, f)
    print("Wrote:", out_nb)


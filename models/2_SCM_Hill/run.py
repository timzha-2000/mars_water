#!/usr/bin/env python
# coding: utf-8

# In[1]:


# run_demni_cases.py
import os, re, copy, nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

# 1) Point this to your DEM notebook (update if your filename differs)
orig_path = "Berryman_mod_wood.ipynb"

# 2) Output pattern + dir
out_tpl    = "bmw_executed_{name}.ipynb"
output_dir = "outputs_bmw"

# 3) Three cases
cases = [
    {
        "name": "A_constraints_away",
        "Vp_kms": 3.8, "Vs_kms": 2.2, "rho": 2589,
        "sigVp_kms": 1.0, "sigVs_kms": 0.4, "sigRho": 157
    },
    {
        "name": "B_wright_inherited",
        "Vp_kms": 4.1, "Vs_kms": 2.5, "rho": 2589,
        "sigVp_kms": 0.2, "sigVs_kms": 0.3, "sigRho": 157
    },
    {
        "name": "C_insight_marsquake",
        "Vp_kms": 4.7, "Vs_kms": 2.7, "rho": 2589,
        "sigVp_kms": 0.3, "sigVs_kms": 0.1, "sigRho": 157
    },
]

# 4) Regex patterns to patch d and s in your notebook
pat_d = re.compile(r"^\s*d\s*=\s*np\.array\([^)]*\)\s*$", re.MULTILINE)
pat_s = re.compile(r"^\s*s\s*=\s*np\.array\([^)]*\)\s*$", re.MULTILINE)

def build_patched_nb(nb, case_name, d_vals, s_vals, output_dir):
    """Create a new notebook with:
       - a preamble (Agg backend + case tagging + savefig monkeypatch)
       - all original cells but with d and s assignments patched via regex
       - a tail cell that saves any open figs AND npy arrays if present
    """
    patched = nbformat.v4.new_notebook()
    patched.metadata = nb.metadata

    preamble = f"""
# --- injected preamble for headless + tagging ---
import os, matplotlib
matplotlib.use('Agg')  # no display
import matplotlib.pyplot as plt

case_name  = {case_name!r}
output_dir = {output_dir!r}
os.makedirs(output_dir, exist_ok=True)

# Monkey-patch: always save into output_dir and tag with case_name
_ORIG_SAVEFIG = plt.Figure.savefig

def _savefig(self, fname, *args, **kwargs):
    import os
    base = os.path.basename(str(fname))
    root, ext = os.path.splitext(base)   # <-- root defined HERE
    if case_name not in root:
        root = f"{{root}}_{{case_name}}"
    out = os.path.join(output_dir, root + (ext or ".png"))
    return _ORIG_SAVEFIG(self, out, *args, **kwargs)

plt.Figure.savefig = _savefig


# Patch plt.savefig too (many notebooks call pyplot.savefig)
_ORIG_PLT_SAVEFIG = plt.savefig

def _plt_savefig(fname, *args, **kwargs):
    import os
    base = os.path.basename(str(fname))
    root, ext = os.path.splitext(base)   # <-- root defined HERE
    if case_name not in root:
        root = f"{{root}}_{{case_name}}"
    out = os.path.join(output_dir, root + (ext or ".png"))
    return _ORIG_PLT_SAVEFIG(out, *args, **kwargs)

plt.savefig = _plt_savefig


print(f">> Running case '{{case_name}}'; outputs -> {{output_dir}}")
"""
    patched.cells.append(nbformat.v4.new_code_cell(preamble))

    # Build the replacement lines
    d_line = f"d = np.array([{d_vals[0]:.6g}, {d_vals[1]:.6g}, {d_vals[2]:.6g}])"
    s_line = f"s = np.array([{s_vals[0]:.6g}, {s_vals[1]:.6g}, {s_vals[2]:.6g}])"

    for cell in nb.cells:
        if cell.cell_type == "code":
            c = copy.deepcopy(cell)
            c.source = pat_d.sub(d_line, c.source)
            c.source = pat_s.sub(s_line, c.source)
            patched.cells.append(c)
        else:
            patched.cells.append(copy.deepcopy(cell))

    tail = r"""
# --- tail: save open figs and .npy arrays if present ---
import os, numpy as np, matplotlib.pyplot as plt

# Save any currently open figures (safety net)
nums = plt.get_fignums()
for i, n in enumerate(nums, 1):
    fig = plt.figure(n)
    out = os.path.join(output_dir, f"{case_name}_fig{i}.png")
    fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close('all')
print(f">> Saved {len(nums)} figures to {output_dir}")

# Helper to save arrays if they exist
def _maybe_save(var_name, out_stem):
    if var_name in globals():
        try:
            arr = np.asarray(globals()[var_name])
            out = os.path.join(output_dir, f"{out_stem}_{case_name}.npy")
            np.save(out, arr)
            print(f">> Saved {os.path.basename(out)} shape={arr.shape}")
        except Exception as e:
            print(f">> Could not save {var_name} -> {out_stem}: {e}")

# Saturation array candidates -> saturation_demni_<CASE>.npy
_maybe_save('idx_water',      'saturation_demni')
_maybe_save('idx_saturation', 'saturation_demni')  # fallback name

# Porosity array candidates -> porosity_demni_<CASE>.npy
_maybe_save('idx_porosity', 'porosity_demni')
_maybe_save('idx_por',      'porosity_demni')      # fallback name
"""
    patched.cells.append(nbformat.v4.new_code_cell(tail))
    return patched

# 5) Read original DEM notebook
nb_orig = nbformat.read(orig_path, as_version=4)

# 6) Filter cases if a case name is provided via CLI
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
    d_vals = (c["Vp_kms"], c["Vs_kms"], c["rho"])
    s_vals = (c["sigVp_kms"], c["sigVs_kms"], c["sigRho"])

    nb_p = build_patched_nb(nb_orig, c["name"], d_vals, s_vals, output_dir)

    try:
        client = NotebookClient(nb_p, timeout=86400, kernel_name="python3")
        client.execute()
    except (CellExecutionError, CellTimeoutError) as e:
        print(f"[{c['name']}] Execution error:\n{e}")

    out_nb = out_tpl.format(name=c["name"])
    with open(out_nb, "w") as f:
        nbformat.write(nb_p, f)
    print("Wrote:", out_nb)


# In[ ]:




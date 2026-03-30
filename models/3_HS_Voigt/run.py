#!/usr/bin/env python
# coding: utf-8

# In[1]:


# run_hsu_cases.py
import os, re, nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# 1) Path to your HS notebook
orig_path = "HS.ipynb"   # update if your filename differs

# 2) Output naming
out_tpl    = "hsu_executed_{name}.ipynb"
output_dir = "outputs_hsu"

# 3) Three cases (Vp, Vs in km/s; rho in kg/m^3; std in same units)
cases = [
    {"name": "A_constraints_away",
     "Vp_kms": 3.8, "Vs_kms": 2.2, "rho": 2589,
     "sigVp_kms": 1.0, "sigVs_kms": 0.4, "sigRho": 157},
    {"name": "B_wright_inherited",
     "Vp_kms": 4.1, "Vs_kms": 2.5, "rho": 2589,
     "sigVp_kms": 0.2, "sigVs_kms": 0.3, "sigRho": 157},
    {"name": "C_insight_marsquake",
     "Vp_kms": 4.7, "Vs_kms": 2.7, "rho": 2589,
     "sigVp_kms": 0.3, "sigVs_kms": 0.1, "sigRho": 157},
]

# 4) Regex patterns for both notebook formats
pat_d = re.compile(r"^\s*d\s*=\s*np\.array\([^)]*\)\s*$", re.MULTILINE)
pat_s = re.compile(r"^\s*s\s*=\s*np\.array\([^)]*\)\s*$", re.MULTILINE)
pat_vp  = re.compile(r"^\s*Vp_obs\s*=\s*[^#\n]+", re.MULTILINE)
pat_vs  = re.compile(r"^\s*Vs_obs\s*=\s*[^#\n]+", re.MULTILINE)
pat_rho = re.compile(r"^\s*rho_obs\s*=\s*[^#\n]+", re.MULTILINE)
pat_svp = re.compile(r"^\s*sigma_Vp\s*=\s*[^#\n]+", re.MULTILINE)
pat_svs = re.compile(r"^\s*sigma_Vs\s*=\s*[^#\n]+", re.MULTILINE)
pat_sro = re.compile(r"^\s*sigma_rho\s*=\s*[^#\n]+", re.MULTILINE)

def build_patched_nb(nb, case_name, d_vals, s_vals, output_dir):
    """Create a notebook that injects case metadata, modifies Vp/Vs/rho, and saves npy outputs."""
    patched = nbformat.v4.new_notebook()
    patched.metadata = nb.metadata

    preamble = f"""
# --- injected preamble (headless execution) ---
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

case_name  = {case_name!r}
output_dir = {output_dir!r}
os.makedirs(output_dir, exist_ok=True)

# Monkey-patch savefig to tag output names
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

    # Build replacement lines
    d_line = f"d = np.array([{d_vals[0]:.6g}, {d_vals[1]:.6g}, {d_vals[2]:.6g}])"
    s_line = f"s = np.array([{s_vals[0]:.6g}, {s_vals[1]:.6g}, {s_vals[2]:.6g}])"
    vp_line  = f"Vp_obs = {d_vals[0]:.6g}"
    vs_line  = f"Vs_obs = {d_vals[1]:.6g}"
    ro_line  = f"rho_obs = {d_vals[2]:.6g}"
    svp_line = f"sigma_Vp = {s_vals[0]:.6g}"
    svs_line = f"sigma_Vs = {s_vals[1]:.6g}"
    sro_line = f"sigma_rho = {s_vals[2]:.6g}"

    for cell in nb.cells:
        if cell.cell_type != "code":
            patched.cells.append(cell)
            continue

        src = cell.source
        new_src = pat_d.sub(d_line, src)
        new_src = pat_s.sub(s_line, new_src)
        new_src = pat_vp.sub(vp_line, new_src)
        new_src = pat_vs.sub(vs_line, new_src)
        new_src = pat_rho.sub(ro_line, new_src)
        new_src = pat_svp.sub(svp_line, new_src)
        new_src = pat_svs.sub(svs_line, new_src)
        new_src = pat_sro.sub(sro_line, new_src)

        patched.cells.append(nbformat.v4.new_code_cell(new_src))

    tail = r"""
# --- tail: save open figures and npy arrays ---
import os, numpy as np, matplotlib.pyplot as plt

# Save figures
nums = plt.get_fignums()
for i, n in enumerate(nums, 1):
    fig = plt.figure(n)
    out = os.path.join(output_dir, f"{case_name}_fig{i}.png")
    fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close('all')
print(f">> Saved {len(nums)} figures to {output_dir}")

# Try saving arrays if defined
def _maybe_save(var_name, stem):
    if var_name in globals():
        try:
            arr = np.asarray(globals()[var_name])
            out = os.path.join(output_dir, f"{stem}_{case_name}.npy")
            np.save(out, arr)
            print(f">> Saved {os.path.basename(out)} shape={arr.shape}")
        except Exception as e:
            print(f">> Could not save {{var_name}}: {{e}}")

_maybe_save("idx_water", "saturation_hs")
_maybe_save("idx_porosity", "porosity_hs")
"""
    patched.cells.append(nbformat.v4.new_code_cell(tail))
    return patched

# 5) Read original HS notebook
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
    d_vals = (c["Vp_kms"], c["Vs_kms"], c["rho"])
    s_vals = (c["sigVp_kms"], c["sigVs_kms"], c["sigRho"])

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


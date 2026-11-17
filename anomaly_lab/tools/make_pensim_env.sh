#!/bin/bash
# Builds PenSimPy's own environment in ./pensim-env (at the repository root; ignored by git), then checks it by
# generating one short batch. Run from the repository root:  bash anomaly_lab/tools/make_pensim_env.sh
set -e
uv venv pensim-env --python 3.12
PY=pensim-env/Scripts/python.exe
[ -f "$PY" ] || PY=pensim-env/bin/python
uv pip install --python "$PY" pensimpy==1.2.5 --no-deps
uv pip install --python "$PY" -r anomaly_lab/tools/pensim-requirements.txt
cd anomaly_lab
PYTHONPATH=. "../$PY" -c "from adlab import pensim; b = pensim.generate(seed=1, hours=10); print('PenSimPy works:', len(b), 'readings')"
echo "Generate the campaign with:  cd anomaly_lab && PYTHONPATH=. ../$PY -m adlab.pensim campaign 8"

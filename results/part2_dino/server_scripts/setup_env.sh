#!/bin/bash
set -ex
P=$HOME/aiaa3201_p2
export CONDA_PKGS_DIRS=$P/.cache/conda_pkgs CONDA_REGISTER_ENVS=false PIP_CACHE_DIR=$P/.cache/pip TMPDIR=$P/.tmp XDG_CACHE_HOME=$P/.cache PYTHONNOUSERSITE=1
/opt/miniconda3/bin/conda --version
/opt/miniconda3/bin/conda create -y -p $P/env --override-channels -c conda-forge python=3.11.13 pip
$P/env/bin/python -m pip install --no-input --extra-index-url https://download.pytorch.org/whl/cu128 -r $P/requirements-server.txt
$P/env/bin/python -m pip install --no-input -r $P/shared_protocol/requirements.txt
$P/env/bin/python -m pip check || true
$P/env/bin/python -c "import cv2;print('cv2',cv2.__version__)" || true
echo ENV_SETUP_DONE

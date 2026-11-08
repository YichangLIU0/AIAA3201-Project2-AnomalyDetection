#!/bin/bash
P=$HOME/aiaa3201_p2; cd $P
export TMPDIR=$P/.tmp XDG_CACHE_HOME=$P/.cache PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
PY=$P/env/bin/python
echo "== check_ready"; $PY -m shared_protocol.check_ready --output-dir $P/logs/check_ready_reports; echo "check_ready exit=$?"
echo "== shared tests"; $PY -m unittest discover -s shared_protocol/tests -t . -v; echo "shared exit=$?"
echo "== part2 tests"; $PY -m unittest part2_official/test_part2_official.py -v; echo "part2 exit=$?"
echo CHECKS_DONE

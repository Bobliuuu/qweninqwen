#!/bin/bash
# usage: run_variant.sh <variant> <out_prefix> [scale] [ngen]   (runs on the GPU box)
V=$1; O=$2; SC=${3:-real151936_L28}; NG=${4:-1}
cd experiments/vslice
python3 audit/nofla_launch.py --variant $V --mode log --fast --entry layer --out out/$O.json \
 --scope weight_prep:vs2.py:tile_payload --scope weight_prep:test_vs2.py:weights \
 --scope build:vs2.py:compile_entry --scope build:vs2.py:build_layout --scope build:vs2.py:make_plan \
 --scope 'build:engine.py:<module>' --scope transport:vs2.py:dense_job \
 test_vs2.py $SC $NG > out/$O.log 2>&1
echo "exit=$?" >> out/$O.log
date -u +%FT%TZ > out/$O.end_utc

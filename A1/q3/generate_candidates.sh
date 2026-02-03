#!/bin/bash
set -e

DB_VEC=$1
Q_VEC=$2
OUT_FILE=$3


python direct_si.py $DB_VEC $Q_VEC $OUT_FILE

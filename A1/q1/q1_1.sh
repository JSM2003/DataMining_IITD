#!/bin/bash
# Usage:
# bash q1_1.sh <path_apriori_executable> <path_fp_executable> <path_dataset> <path_out>

set -o pipefail

APRIORI=$1
FPGROWTH=$2
DATASET=$3
OUTDIR=$4

SUPPORTS=(5 10 25 50 90)
TIMEOUT=3600   # 1 hour

mkdir -p "$OUTDIR"

# Bash built-in time format (real time in seconds)
TIMEFORMAT='%3R'

echo "=============================================="
echo "Q1 Task 1: Apriori vs FP-Growth Runtime Study"
echo "=============================================="

############################
# Run Apriori
############################
for S in "${SUPPORTS[@]}"; do
    echo ""
    echo "Running Apriori at ${S}% support"
    mkdir -p "$OUTDIR/ap${S}"

    {
        time timeout $TIMEOUT \
            "$APRIORI" \
            -o \
            -s${S} \
            -v"," \
            "$DATASET" \
            "$OUTDIR/ap${S}/output.txt" \
            2> "$OUTDIR/ap${S}/program.err"
    } 2> "$OUTDIR/ap${S}/time.txt"

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 124 ]; then
        echo "TIMEOUT" > "$OUTDIR/ap${S}/time.txt"
        echo "Apriori timed out at ${S}% support" > "$OUTDIR/ap${S}/output.txt"
    else
        # keep only numeric time
        sed -i '$!d' "$OUTDIR/ap${S}/time.txt"
    fi
done

############################
# Run FP-Growth
############################
for S in "${SUPPORTS[@]}"; do
    echo ""
    echo "Running FP-Growth at ${S}% support"
    mkdir -p "$OUTDIR/fp${S}"

    FP_S=$(echo "scale=2; $S/100" | bc)

    {
        time timeout $TIMEOUT \
            "$FPGROWTH" \
            -s${FP_S} \
            -o \
            -v"," \
            "$DATASET" \
            "$OUTDIR/fp${S}/output.txt" \
            2> "$OUTDIR/fp${S}/program.err"
    } 2> "$OUTDIR/fp${S}/time.txt"

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 124 ]; then
        echo "TIMEOUT" > "$OUTDIR/fp${S}/time.txt"
        echo "FP-Growth timed out at ${S}% support" > "$OUTDIR/fp${S}/output.txt"
    else
        sed -i '$!d' "$OUTDIR/fp${S}/time.txt"
    fi
done

############################
# Generate plot
############################
echo ""
echo "Generating runtime plot..."
python3 plot.py "$OUTDIR"

echo ""
echo "Q1 Task 1 completed successfully."

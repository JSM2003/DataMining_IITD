PATH_GSPAN_EXEC="$1"
PATH_FSG_EXEC="$2"
PATH_GASTON_EXEC="$3"
RAW_DATASET_PATH="$4"
PATH_OUT="$5"

PROCESSED_INPUT_PATH="processed_input"

[ -d "$PROCESSED_INPUT_PATH" ] && rm -r "$PROCESSED_INPUT_PATH"
mkdir -p "$PROCESSED_INPUT_PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROCESSED_INPUT_FILE_GLIB="${SCRIPT_DIR}/${PROCESSED_INPUT_PATH}/formatted_ip_gspan_gaston"
PROCESSED_INPUT_FILE_FSG="${SCRIPT_DIR}/${PROCESSED_INPUT_PATH}/formatted_ip_fsg"

# ────────────────────────────────────────────────
# Function: get current timestamp
# ────────────────────────────────────────────────
timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

echo "────────────────────────────────────────────"
echo "[INFO] Starting data preprocessing…"
echo "[TIME] $(timestamp)"

# ────────────────────────────────────────────────
# Run Python preprocessing
# ────────────────────────────────────────────────

echo "[INFO] Preprocessing- Gspan, Gaston"
start_preproc_glib=$(date +%s)

python3 "$SCRIPT_DIR/preproc_scripts/gspan_gaston_data_adaptor.py" "$RAW_DATASET_PATH" "$PROCESSED_INPUT_FILE_GLIB"

preproc_time_glib=$(($(date +%s) - start_preproc_glib))
echo "[INFO] Preprocessing completed- Gspan, Gaston, TT: ${preproc_time_glib} seconds"

echo "[INFO] Preprocessing- FSG"
start_preproc_fsg=$(date +%s)

python3 "$SCRIPT_DIR/preproc_scripts/fsg_data_adaptor.py" "$RAW_DATASET_PATH" "$PROCESSED_INPUT_FILE_FSG"

preproc_time_fsg=$(($(date +%s) - start_preproc_fsg))
echo "[INFO] Preprocessing completed- FSG, TT: ${preproc_time_fsg} seconds"

total_preproc_time=$(( preproc_time_glib + preproc_time_fsg ))

echo "[INFO] Data preprocessing completed."
echo "[TIME] $(timestamp)"
echo "[TT] Total Preprocessing Time: ${total_preproc_time} seconds"
echo "────────────────────────────────────────────"
echo

# ────────────────────────────────────────────────
# Run Subgraph Mining Algorithm
# ────────────────────────────────────────────────
echo "────────────────────────────────────────────"
echo "[INFO] Starting subgraph mining…"
echo "[TIME] $(timestamp)"

THRESHOLDS="5 10 25 50 95"
TOTAL_TRANS=$(grep -c "#" raw_dataset/yeast_167.txt_graph)
echo "Total no of graph transactions: ${TOTAL_TRANS}"

[ -d "$PATH_OUT" ] && rm -r "$PATH_OUT"
mkdir -p "$PATH_OUT"

execution_results_file="output/results.csv"
echo "algorithm,threshold,exec_time" > "$execution_results_file"

start_time=$(date +%s)

for t in $THRESHOLDS; do

  echo "[INFO] Executing for threshold: ${t}"
  echo "[INFO] Starting Gspan"
  # convert integer threshold to float for gspan
  float_t=$(printf "0.%02d" "$t")
  start_gspan=$(date +%s)

  "$PATH_GSPAN_EXEC" -f "$PROCESSED_INPUT_FILE_GLIB" -s "$float_t" -o
  mv "${PROCESSED_INPUT_FILE_GLIB}.fp" "${PATH_OUT}/gspan${t}.fp" 2>/dev/null

  gspan_tt=$(($(date +%s) - start_gspan))
  echo "gspan,$t,$gspan_tt" >> "$execution_results_file"
  echo "[INFO] Completed gspan threshold $t, TT: ${gspan_tt} seconds"
  echo

  echo "[INFO] Starting FSG"
  start_fsg=$(date +%s)

  "$PATH_FSG_EXEC" -s "$t" "$PROCESSED_INPUT_FILE_FSG"
  mv "${PROCESSED_INPUT_FILE_FSG}.fp" "${PATH_OUT}/fsg${t}.fp" 2>/dev/null

  fsg_tt=$(($(date +%s) - start_fsg))
  echo "fsg,$t,$fsg_tt" >> "$execution_results_file"
  echo "[INFO] Completed fsg threshold $t, TT: ${fsg_tt} seconds"
  echo

  echo "[INFO] Starting Gaston"
  start_gaston=$(date +%s)
  gaston_op_file="output/gaston${t}.fp"

  min_sup_gaston=$(echo "scale=6; $TOTAL_TRANS * $float_t" | bc | cut -d. -f1)
  "$PATH_GASTON_EXEC" "$min_sup_gaston" "$PROCESSED_INPUT_FILE_GLIB" "$gaston_op_file"

  gaston_tt=$(($(date +%s) - start_gaston))
  echo "gaston,$t,$gaston_tt" >> "$execution_results_file"
  echo "[INFO] Completed gaston threshold $t, TT: ${gaston_tt} seconds"
  echo
done

total_time=$(($(date +%s) - start_time))

echo "[INFO] Subgraph mining completed."
echo "[TIME] $(timestamp)"
echo "[TT] Total Execution Time: ${total_time} seconds"
echo "────────────────────────────────────────────"
echo

# ────────────────────────────────────────────────
# Plot the results
# ────────────────────────────────────────────────

echo "[INFO] Plotting execution results"

python3 "$SCRIPT_DIR/plot_scripts/results_plot.py" "$execution_results_file"

echo "[INFO] Completed plotting the results"
echo


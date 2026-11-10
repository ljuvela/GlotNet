#!/usr/bin/env bash
# Run the whole smoke test on one machine (laptop / workstation / interactive node).
#
# usage: run_local.sh UTTERANCE.wav OUT_DIR [ITERS=5000] [PARALLEL=2] [THREADS=4] [DEVICE=cpu]
#
# Trains every run in runs.py (PARALLEL at a time, THREADS each), then
# evaluates them all and runs the voicing check. Logs: OUT_DIR/train_<run>.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
UTT="$1"; OUT="$2"; ITERS="${3:-5000}"; PAR="${4:-2}"; THREADS="${5:-4}"; DEVICE="${6:-cpu}"
mkdir -p "$OUT"

python -I "$HERE/train.py" --prepare-only --utterance "$UTT" --out "$OUT"
# PAR runs at a time (polling: macOS bash 3.2 has no `wait -n`)
for RUN in $(python -I "$HERE/runs.py"); do
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 2; done
    echo "start: $RUN"
    python -I "$HERE/train.py" --run "$RUN" --utterance "$UTT" --out "$OUT" \
        --iters "$ITERS" --threads "$THREADS" --device "$DEVICE" \
        > "$OUT/train_$RUN.log" 2>&1 &
done
wait
FAILED=0
for RUN in $(python -I "$HERE/runs.py"); do
    if [ -f "$OUT/$RUN/run.json" ]; then echo "done: $RUN"
    else echo "FAILED: $RUN (see $OUT/train_$RUN.log)"; FAILED=1; fi
done
[ "$FAILED" -eq 0 ] || exit 1

python -I "$HERE/evaluate.py" "$OUT" | tee "$OUT/evaluation.log"
python -I "$HERE/voicing_check.py" "$OUT" gn_gl_shared | tee "$OUT/voicing_check.log"

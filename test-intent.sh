#!/bin/bash
# intent-replica on Triton (python backend): utterance in, best corpus match out.
#
# Reproduces the production intent-classifier shape -- BERT encode plus a
# cosine search over a corpus that lives on the server, so the request carries
# only the user utterance and the response carries the matched line back.
#
# The corpus is 36,912 rows from banking77 + CLINC150 (see
# triton-models/intent-replica/tools/build_corpus.py). Scores are cosine
# similarity against unit-length vectors, so 1.0 is an exact match.
#
#   ./test-intent.sh                  # one freeform utterance
#   ./test-intent.sh "your text"      # any utterance you like
#   ./test-intent.sh --check          # regression check, exits non-zero on failure

HOST=http://192.168.1.151:8000
MODEL=intent-replica

infer() {
  curl -s -X POST $HOST/v2/models/$MODEL/infer \
    -H "Content-Type: application/json" \
    -d "{\"inputs\":[{\"name\":\"TEXT\",\"datatype\":\"BYTES\",\"shape\":[1,1],\"data\":[\"$1\"]}]}"
}

if [ "$1" = "--check" ]; then
  # A known corpus line must return itself at score 1.0. This is the gate that
  # catches a silent argmax or alignment bug, which would otherwise return
  # plausible strings at plausible latency.
  QUERY='How do I know if I will get my card, or if it is lost?'
  infer "$QUERY" | python3 -c '
import json, sys
d = json.load(sys.stdin)
if "error" in d:
    sys.exit(d["error"])
out = {o["name"]: o["data"][0] for o in d["outputs"]}
print("MATCH :", out["MATCH"])
print("INTENT:", out["INTENT"])
print("SCORE :", out["SCORE"])
if out["MATCH"] != sys.argv[1] or abs(out["SCORE"] - 1.0) > 1e-4:
    sys.exit("FAIL: expected a self-match at score 1.0")
print("PASS")
' "$QUERY"
  exit $?
fi

# Batched form works too -- shape [N,1] returns one row per input.
TEXT="${1:-I want to reset the password on my account}"
infer "$TEXT" | python3 -c '
import json, sys
d = json.load(sys.stdin)
if "error" in d:
    sys.exit(d["error"])
out = {o["name"]: o["data"][0] for o in d["outputs"]}
print("MATCH :", out["MATCH"])
print("INTENT:", out["INTENT"])
print("SCORE :", round(out["SCORE"], 4))
'

#!/bin/bash
# Stella 400M on Triton (python backend): text in, unit-length 1024-d embedding out.
#
# This is the encoder half of the production intent shape. The corpus itself
# lives in intent-replica; stella400m just turns a string into a vector.
#
# Stella is an ASYMMETRIC model: the instruct prompt goes on queries and must
# NOT be used on passages. The server prepends it by default (PROMPT_MODE=always
# in config.pbtxt), so a plain request is a query. Pass --passage to set
# USE_PROMPT=0 and get a document-side vector instead.
#
#   ./test-stella.sh                      # one freeform string, query side
#   ./test-stella.sh "your text"          # any string you like
#   ./test-stella.sh --passage "a doc"    # document side (no instruct prompt)
#   ./test-stella.sh --check              # regression gate, non-zero on failure

HOST=${HOST:-http://192.168.1.151:8000}
MODEL=stella400m

# Sends TEXT with the given USE_PROMPT flag and prints the raw JSON.
infer() {
  local text="$1" prompt="$2"
  curl -s -X POST $HOST/v2/models/$MODEL/infer \
    -H "Content-Type: application/json" \
    -d "{\"inputs\":[
          {\"name\":\"TEXT\",\"datatype\":\"BYTES\",\"shape\":[1,1],\"data\":[\"$text\"]},
          {\"name\":\"USE_PROMPT\",\"datatype\":\"BOOL\",\"shape\":[1,1],\"data\":[$prompt]}
        ]}"
}

if [ "$1" = "--check" ]; then
  # Stella is only worth running if it can tell a matching passage from an
  # unrelated one. Both bugs that break that here (uninitialised rope buffers
  # under transformers 5, a mis-applied instruct prompt) are silent: the model
  # still returns unit-length vectors, they are just nearly identical for every
  # input. So the gate is a ranking test, not a shape test.
  QUERY="What is the capital of France?"
  GOOD="Paris is the capital and largest city of France."
  BAD="My cat likes to sleep on the couch all day."

  vec() {
    infer "$1" "$2" | python3 -c '
import json, sys
raw = sys.stdin.read()
if not raw.strip():
    # A slow or lossy link (wifi to the box) yields an empty body from curl -s.
    # Say so instead of drowning in a JSONDecodeError traceback.
    sys.exit("ERROR: empty response from server (check the link, or run this "
             "on the host with HOST=http://localhost:8000)")
d = json.loads(raw)
if "error" in d:
    sys.exit(d["error"])
o = [x for x in d["outputs"] if x["name"] == "EMBEDDING"][0]
print(" ".join(map(str, o["data"])))
'
  }
  Q=$(vec "$QUERY" true)
  G=$(vec "$GOOD" false)
  B=$(vec "$BAD" false)

  python3 - "$Q" "$G" "$B" "$GOOD" "$BAD" <<'PY'
import sys, math

def v(s):
    xs = [float(x) for x in s.split()]
    n = math.sqrt(sum(x * x for x in xs))
    return xs, n

q, qn = v(sys.argv[1])
g, gn = v(sys.argv[2])
b, bn = v(sys.argv[3])

for name, vec, norm in (("query", q, qn), ("passage", g, gn), ("passage", b, bn)):
    if len(vec) != 1024:
        sys.exit(f"FAIL: expected 1024 dims, got {len(vec)}")
    if abs(norm - 1.0) > 1e-3:
        sys.exit(f"FAIL: {name} is not unit length (norm={norm:.6f})")

dot = lambda a, c: sum(x * y for x, y in zip(a, c))
sg, sb = dot(q, g), dot(q, b)
print(f"dims  : 1024  (unit norm ok)")
print(f"match : {sg:.4f}   {sys.argv[4]}")
print(f"apart : {sb:.4f}   {sys.argv[5]}")
print(f"gap   : {sg - sb:.4f}")

# Collapse shows up as a gap near zero. 0.2 is comfortably above the noise and
# comfortably below what the healthy model actually scores (~0.6 here).
if sg - sb < 0.2:
    sys.exit("FAIL: similarity gap too small -- embeddings look collapsed")
if not sg > sb:
    sys.exit("FAIL: ranking inverted -- unrelated text scored highest")
print("PASS")
PY
  exit $?
fi

PROMPT=true
if [ "$1" = "--passage" ]; then
  PROMPT=false
  shift
fi
TEXT="${1:-I want to reset the password on my account}"

infer "$TEXT" "$PROMPT" | python3 -c '
import json, math, sys
raw = sys.stdin.read()
if not raw.strip():
    sys.exit("ERROR: empty response from server (check the link, or run this "
             "on the host with HOST=http://localhost:8000)")
d = json.loads(raw)
if "error" in d:
    sys.exit(d["error"])
o = [x for x in d["outputs"] if x["name"] == "EMBEDDING"][0]
v = o["data"]
norm = math.sqrt(sum(x * x for x in v))
print("SHAPE :", "x".join(map(str, o["shape"])), "->", len(v), "values")
print("NORM  :", round(norm, 6))
print("FIRST8:", [round(x, 4) for x in v[:8]])
'

#!/usr/bin/env bash
# Task 4: static analysis.
#   Bandit    - Python SAST (listed on OWASP's Source Code Analysis Tools page)
#   Semgrep   - rule-based SAST: general Python, security-audit, Trail of Bits (ML-specific) and secrets rules
#   pip-audit - known-vulnerable dependencies (SCA; complements SAST for OWASP ML06 supply chain)
# The application code and the control sample (sast/control/, deliberately insecure patterns) are
# scanned separately. Reports go to results/sast/. The tools exit non-zero when they find something,
# so failures are tolerated and every tool still runs.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=results/sast
mkdir -p "$OUT"
TARGETS=(src attacks defenses)
SEMGREP_RULES=(--config p/python --config p/security-audit --config p/trailofbits --config p/secrets)

echo "== Bandit =="
bandit -r "${TARGETS[@]}" -f txt -o "$OUT/bandit.txt"
bandit -r "${TARGETS[@]}" -f json -o "$OUT/bandit.json" -q
bandit sast/control -r -f txt -o "$OUT/bandit_control.txt" -q

echo "== Semgrep =="
semgrep scan "${SEMGREP_RULES[@]}" --metrics=off --output "$OUT/semgrep.txt" "${TARGETS[@]}"
semgrep scan "${SEMGREP_RULES[@]}" --metrics=off --output "$OUT/semgrep_control.txt" sast/control

echo "== pip-audit =="
pip-audit -r requirements.txt -f markdown -o "$OUT/pip-audit.md"

echo "Reports written to $OUT/"

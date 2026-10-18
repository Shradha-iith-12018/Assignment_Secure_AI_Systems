#!/usr/bin/env bash
# Task 4: SAST + dependency audit.
#
# Application findings are recorded but do not stop the script, so all reports
# are always produced. A deliberately insecure control sample is scanned
# separately to verify that the configured tools/rules detect known problems.

set -uo pipefail

cd "$(dirname "$0")/.."

OUT="results/sast"
mkdir -p "$OUT"

TARGETS=(src attacks defenses scripts)

SEMGREP_RULES=(
    --config p/python
    --config p/security-audit
    --config p/trailofbits
    --config p/secrets
)

SEMGREP_COMMON=(
    scan
    "${SEMGREP_RULES[@]}"
    --metrics=off
    --no-git-ignore
    --error
)

STATUS_FILE="$OUT/status.txt"
: > "$STATUS_FILE"


record_status() {
    local label="$1"
    local code="$2"
    printf '%s=%s\n' "$label" "$code" >> "$STATUS_FILE"
}


echo "== Tool versions =="

{
    echo "bandit: $(bandit --version 2>&1 | head -n 1)"
    echo "semgrep: $(semgrep --version 2>&1 | head -n 1)"
    echo "pip-audit: $(pip-audit --version 2>&1 | head -n 1)"
} > "$OUT/tool_versions.txt"


echo "== Bandit: application =="

bandit \
    -r "${TARGETS[@]}" \
    -f txt \
    -o "$OUT/bandit.txt"
record_status bandit_app_txt "$?"

bandit \
    -r "${TARGETS[@]}" \
    -f json \
    -o "$OUT/bandit.json" \
    -q
record_status bandit_app_json "$?"


echo "== Bandit: deliberately insecure control sample =="

bandit \
    -r sast/control \
    -f txt \
    -o "$OUT/bandit_control.txt" \
    -q
record_status bandit_control_txt "$?"

bandit \
    -r sast/control \
    -f json \
    -o "$OUT/bandit_control.json" \
    -q
record_status bandit_control_json "$?"


echo "== Semgrep: application =="

semgrep \
    "${SEMGREP_COMMON[@]}" \
    --output "$OUT/semgrep.txt" \
    "${TARGETS[@]}"
record_status semgrep_app_txt "$?"

semgrep \
    "${SEMGREP_COMMON[@]}" \
    --json \
    --output "$OUT/semgrep.json" \
    "${TARGETS[@]}"
record_status semgrep_app_json "$?"


echo "== Semgrep: deliberately insecure control sample =="

semgrep \
    "${SEMGREP_COMMON[@]}" \
    --output "$OUT/semgrep_control.txt" \
    sast/control
record_status semgrep_control_txt "$?"

semgrep \
    "${SEMGREP_COMMON[@]}" \
    --json \
    --output "$OUT/semgrep_control.json" \
    sast/control
record_status semgrep_control_json "$?"


echo "== pip-audit =="

pip-audit \
    -r requirements.txt \
    -f json \
    -o "$OUT/pip-audit.json"
record_status pip_audit_json "$?"

pip-audit \
    -r requirements.txt \
    > "$OUT/pip-audit.txt" 2>&1
record_status pip_audit_txt "$?"


echo
echo "Reports written to $OUT/"
echo "Exit statuses written to $STATUS_FILE."
echo
echo "Interpretation:"
echo "  application status 0 = no findings / successful scan"
echo "  control status non-zero = expected because insecure examples are intentional"

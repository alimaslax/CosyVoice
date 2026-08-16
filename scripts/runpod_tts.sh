#!/usr/bin/env bash
# Usage: ./scripts/runpod_tts.sh <slow|medium|fast> "Somali text" [output.wav]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
# Default to the dedicated RTX 4090 endpoint. Set RUNPOD_ENDPOINT_ID to
# override this per call without changing the script.
ENDPOINT_ID="${RUNPOD_ENDPOINT_ID:-rkaset6oi127wh}"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <slow|medium|fast> \"text\" [output.wav]" >&2
  exit 64
fi

PACE="$1"
TEXT="$2"
OUTPUT_PATH="${3:-${ROOT_DIR}/outputs/runpod-tts-$(date +%Y%m%d-%H%M%S).wav}"

case "$PACE" in slow|medium|fast) ;; *) echo "pace must be slow, medium, or fast" >&2; exit 64 ;; esac

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing ${ENV_FILE}. Copy .env.example to .env and set RUNPOD_API_KEY." >&2
  exit 78
fi

# shellcheck disable=SC1090
source "$ENV_FILE"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is required in .env}"

command -v curl >/dev/null || { echo "curl is required" >&2; exit 69; }
command -v jq >/dev/null || { echo "jq is required (brew install jq)" >&2; exit 69; }

payload="$(jq -n --arg text "$TEXT" --arg pace "$PACE" '{input: {text: $text, pace: $pace}}')"
job_id="$(curl --fail --silent --show-error \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H 'Content-Type: application/json' \
  --data "$payload" \
  "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" | jq -er '.id')"

echo "Runpod job: ${job_id}"
mkdir -p "$(dirname "$OUTPUT_PATH")"
queued_at="$(date +%s)"
processing_at=""

while true; do
  response="$(curl --fail --silent --show-error \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    "https://api.runpod.ai/v2/${ENDPOINT_ID}/status/${job_id}")"
  status="$(jq -r '.status' <<<"$response")"

  case "$status" in
    COMPLETED)
      jq -er '.output.audio_base64' <<<"$response" | base64 -D > "$OUTPUT_PATH"
      delay_ms="$(jq -r '.delayTime // empty' <<<"$response")"
      execution_ms="$(jq -r '.executionTime // empty' <<<"$response")"
      synthesis_seconds="$(jq -r '.output.seconds // empty' <<<"$response")"
      if [[ -n "$delay_ms" ]]; then
        printf 'Queue time: %.3fs\n' "$(awk -v ms="$delay_ms" 'BEGIN { print ms / 1000 }')"
      fi
      if [[ -n "$execution_ms" ]]; then
        printf 'Processing time: %.3fs\n' "$(awk -v ms="$execution_ms" 'BEGIN { print ms / 1000 }')"
      fi
      if [[ -n "$synthesis_seconds" ]]; then
        printf 'Model synthesis time: %.3fs\n' "$synthesis_seconds"
      fi
      echo "Saved WAV: ${OUTPUT_PATH}"
      exit 0
      ;;
    FAILED|CANCELLED|TIMED_OUT)
      jq -r '.error // "Runpod job failed"' <<<"$response" >&2
      exit 1
      ;;
    *)
      if [[ "$status" == "IN_PROGRESS" && -z "$processing_at" ]]; then
        processing_at="$(date +%s)"
        echo "Status: IN_PROGRESS (queue wait: $((processing_at - queued_at))s)"
      elif [[ "$status" == "IN_QUEUE" ]]; then
        echo "Status: IN_QUEUE (waiting: $(( $(date +%s) - queued_at ))s)"
      else
        echo "Status: ${status}"
      fi
      sleep 2
      ;;
  esac
done

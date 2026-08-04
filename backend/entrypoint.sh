#!/bin/sh
set -e

MODE_ALEMBIC="${RUN_ALEMBIC_MIGRATIONS:-blocking}"
MODE_WHISPER="${PRELOAD_WHISPER_MODEL:-false}"

run_alembic() {
  echo "Running Alembic migrations (mode=${MODE_ALEMBIC})..."
  timeout_s="${ALEMBIC_TIMEOUT_SECONDS:-30}"

  for i in 1 2 3 4 5; do
    if timeout "${timeout_s}" alembic -c /app/alembic.ini upgrade head; then
      echo "Alembic migrations complete"
      return 0
    fi
    echo "Alembic failed or timed out (attempt ${i}). Retrying in 2s..."
    sleep 2
  done

  echo "Alembic migrations failed after retries (continuing)"
  return 1
}

run_whisper_preload() {
  echo "Preloading Whisper model (mode=${MODE_WHISPER})..."
  timeout_s="${WHISPER_PRELOAD_TIMEOUT_SECONDS:-600}"
  timeout "${timeout_s}" python /app/preload_models.py --model-size "${WHISPER_MODEL_SIZE:-small}" \
    || echo "Model preload failed/timed out (will download on first use)"
}

# Run Alembic migrations. Alembic is the only source of schema truth, so a
# failed migration means the app would run against a schema it does not expect.
# - "blocking" (default): run first; abort startup if they fail
# - "true": run in background, failures logged only (legacy, unsafe)
# - "false": skip, for environments that migrate out of band
if [ "${MODE_ALEMBIC}" = "blocking" ]; then
  if ! run_alembic; then
    echo "FATAL: Alembic migrations failed; refusing to start against an unknown schema." >&2
    exit 1
  fi
elif [ "${MODE_ALEMBIC}" = "true" ]; then
  run_alembic || true &
fi

# Preload Whisper model.
# - "blocking": run before starting server
# - "true": run in background
# - "false": skip
if [ "${MODE_WHISPER}" = "blocking" ]; then
  run_whisper_preload || true
elif [ "${MODE_WHISPER}" = "true" ]; then
  run_whisper_preload || true &
fi

echo "Starting API server..."
exec "$@"

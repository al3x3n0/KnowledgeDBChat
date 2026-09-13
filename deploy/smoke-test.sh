#!/usr/bin/env bash
#
# Live smoke test for the Helm chart against whatever cluster kubectl points at.
# Installs into a throwaway namespace with lightweight stand-ins for the
# application images (see ci/values-smoke.yaml) and asserts the wiring that
# rendering cannot check.
#
#   ./deploy/smoke-test.sh              # install, assert, uninstall
#   KEEP=1 ./deploy/smoke-test.sh       # leave the release up for inspection
#   NAMESPACE=foo RELEASE=bar ./deploy/smoke-test.sh
#
# Every assertion here corresponds to something that actually broke during
# development, so none of them are ceremonial:
#   - the migration hook running before its ServiceAccount / its database exists
#   - Secret passwords reaching the app only via $(VAR) expansion
#   - the gateway dropping the path when proxying to MinIO
#   - a failed migration silently letting an upgrade roll new pods
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHART_DIR="${REPO_ROOT}/deploy/helm/knowledgedbchat"
VALUES="${CHART_DIR}/ci/values-smoke.yaml"

NAMESPACE="${NAMESPACE:-kdbc-smoke}"
RELEASE="${RELEASE:-kdbc}"
TIMEOUT="${TIMEOUT:-10m}"
FULLNAME="${RELEASE}-knowledgedbchat"

# Must match ci/values-smoke.yaml.
PG_PASSWORD="pgP4ss-w0rd_xyz"
REDIS_PASSWORD="redisP4ss-xyz"
MINIO_USER="smokeminio"
MINIO_PASSWORD="smoke-minio-secret"

PASS=0
FAIL=0
pass() { printf '  \033[0;32mPASS\033[0m %s\n' "$1"; PASS=$((PASS + 1)); }
fail() { printf '  \033[0;31mFAIL\033[0m %s\n' "$1"; FAIL=$((FAIL + 1)); }
step() { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }

# Assert that "$2" (actual) contains "$3" (expected), labelled "$1".
assert_contains() {
  if printf '%s' "$2" | grep -qF -- "$3"; then pass "$1"; else
    fail "$1"
    printf '       expected to contain: %s\n       got: %s\n' "$3" "$2"
  fi
}

cleanup() {
  local rc=$?
  if [[ "${KEEP:-0}" == "1" ]]; then
    printf '\nKEEP=1 — release left at %s/%s\n' "$NAMESPACE" "$RELEASE"
  else
    step "Cleaning up"
    helm uninstall "$RELEASE" -n "$NAMESPACE" --wait >/dev/null 2>&1 || true
    kubectl delete namespace "$NAMESPACE" --wait=false >/dev/null 2>&1 || true
  fi
  exit $rc
}
trap cleanup EXIT

kubectl cluster-info >/dev/null 2>&1 || { echo "no reachable cluster (kubectl cluster-info failed)" >&2; exit 1; }

step "Installing $RELEASE into $NAMESPACE"
# The values file targets minikube's "standard" class; override for clusters
# that name their default differently.
SC_ARGS=()
[[ -n "${SMOKE_STORAGE_CLASS:-}" ]] && SC_ARGS=(--set "global.storageClass=${SMOKE_STORAGE_CLASS}")
helm install "$RELEASE" "$CHART_DIR" -n "$NAMESPACE" --create-namespace \
  -f "$VALUES" "${SC_ARGS[@]}" --timeout "$TIMEOUT" --wait
pass "helm install succeeded (post-install migration hook completed)"

step "Pod and volume state"
kubectl -n "$NAMESPACE" get pods,pvc
UNBOUND=$(kubectl -n "$NAMESPACE" get pvc -o jsonpath='{range .items[*]}{.metadata.name}={.status.phase} {end}' | tr ' ' '\n' | grep -v '=Bound' | grep -c '=' || true)
[[ "$UNBOUND" == "0" ]] && pass "every PVC is Bound (dependencies + the shared /app/data claim)" \
                        || fail "$UNBOUND PVC(s) not Bound"

step "Connection URLs are assembled from the Secret via \$(VAR) expansion"
ENV_OUT=$(kubectl -n "$NAMESPACE" exec "deploy/${FULLNAME}-backend" -- \
  sh -c 'echo "$DATABASE_URL"; echo "$REDIS_URL"; echo "RUN_ALEMBIC_MIGRATIONS=$RUN_ALEMBIC_MIGRATIONS"; echo "VECTOR_STORE_PROVIDER=$VECTOR_STORE_PROVIDER"')
assert_contains "DATABASE_URL carries the Secret password" "$ENV_OUT" ":${PG_PASSWORD}@"
assert_contains "REDIS_URL carries the Secret password"    "$ENV_OUT" ":${REDIS_PASSWORD}@"
assert_contains "app pods never run migrations themselves" "$ENV_OUT" "RUN_ALEMBIC_MIGRATIONS=false"
assert_contains "ConfigMap values arrive via envFrom"      "$ENV_OUT" "VECTOR_STORE_PROVIDER=qdrant"

step "Datastores accept those credentials"
PG_OUT=$(kubectl -n "$NAMESPACE" exec "${FULLNAME}-postgres-0" -- \
  psql "postgresql://user:${PG_PASSWORD}@localhost:5432/knowledge_db" -tAc 'select 1' 2>/dev/null || true)
assert_contains "Postgres authenticates with the assembled URL" "$PG_OUT" "1"

REDIS_NOAUTH=$(kubectl -n "$NAMESPACE" exec "${FULLNAME}-redis-0" -- redis-cli ping 2>&1 || true)
assert_contains "Redis rejects unauthenticated clients" "$REDIS_NOAUTH" "NOAUTH"
REDIS_AUTH=$(kubectl -n "$NAMESPACE" exec "${FULLNAME}-redis-0" -- \
  redis-cli -a "$REDIS_PASSWORD" --no-auth-warning ping 2>/dev/null || true)
assert_contains "Redis accepts the chart's password" "$REDIS_AUTH" "PONG"

QDRANT_OUT=$(kubectl -n "$NAMESPACE" exec "deploy/${FULLNAME}-backend" -- python -c \
  "import urllib.request;print(urllib.request.urlopen('http://${FULLNAME}-qdrant:6333/readyz').read().decode())" 2>/dev/null || true)
assert_contains "Qdrant is reachable over its Service" "$QDRANT_OUT" "ready"

step "MinIO bucket was created by the post-install hook"
MC_OUT=$(kubectl -n "$NAMESPACE" run mc-smoke-$RANDOM --rm -i --restart=Never --quiet \
  --image=minio/mc:RELEASE.2024-10-08T09-37-26Z \
  --overrides="{\"spec\":{\"containers\":[{\"name\":\"mc\",\"image\":\"minio/mc:RELEASE.2024-10-08T09-37-26Z\",\"command\":[\"sh\",\"-c\",\"mc alias set k http://${FULLNAME}-minio:9000 ${MINIO_USER} ${MINIO_PASSWORD} >/dev/null && mc ls k\"],\"stdin\":false}]}}" 2>&1 || true)
assert_contains "documents bucket exists" "$MC_OUT" "documents"

step "Gateway routing"
kubectl -n "$NAMESPACE" port-forward "svc/${FULLNAME}-gateway" 18120:80 >/dev/null 2>&1 &
PF_PID=$!
# shellcheck disable=SC2064
trap "kill $PF_PID 2>/dev/null || true; cleanup" EXIT
for _ in $(seq 1 60); do curl -sf http://localhost:18120/health >/dev/null 2>&1 && break; sleep 1; done

assert_contains "gateway /health"            "$(curl -s http://localhost:18120/health)" "healthy"
assert_contains "/api/ reaches the backend with the path intact" \
  "$(curl -s http://localhost:18120/api/v1/some/path)" "ok /api/v1/some/path"
assert_contains "/ reaches the frontend" \
  "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:18120/)" "200"
# The regression that matters: `proxy_pass $var/` used to drop the path, so
# every object request arrived at MinIO as "/". MinIO echoes the resource it
# was actually asked for.
assert_contains "/minio/ preserves the object path" \
  "$(curl -s http://localhost:18120/minio/documents/)" "<Resource>/documents/</Resource>"

kill $PF_PID 2>/dev/null || true
trap cleanup EXIT

step "Chart's own test suite"
if helm test "$RELEASE" -n "$NAMESPACE" >/dev/null 2>&1; then
  pass "helm test succeeded"
else
  fail "helm test failed"
  kubectl -n "$NAMESPACE" logs "${FULLNAME}-test-connection" 2>/dev/null | tail -20
fi

step "A failed migration must gate the upgrade"
RS_BEFORE=$(kubectl -n "$NAMESPACE" get rs -l app.kubernetes.io/component=backend \
  -o jsonpath='{range .items[?(@.spec.replicas>0)]}{.metadata.name}{end}')
if helm upgrade "$RELEASE" "$CHART_DIR" -n "$NAMESPACE" -f "$VALUES" "${SC_ARGS[@]}" \
     --set-json 'migrations.command=["sh","-c","echo SIMULATED FAILURE >&2; exit 1"]' \
     --set migrations.backoffLimit=0 --set migrations.activeDeadlineSeconds=120 \
     --set config.logLevel=DEBUG --timeout 4m --wait >/dev/null 2>&1; then
  fail "upgrade succeeded despite a failing migration"
else
  pass "upgrade aborted by the failing pre-upgrade migration"
fi
RS_AFTER=$(kubectl -n "$NAMESPACE" get rs -l app.kubernetes.io/component=backend \
  -o jsonpath='{range .items[?(@.spec.replicas>0)]}{.metadata.name}{end}')
[[ "$RS_BEFORE" == "$RS_AFTER" ]] && pass "no new pods rolled while migrations were failing" \
                                  || fail "backend rolled to $RS_AFTER despite the migration failure"
LOG_LEVEL_NOW=$(kubectl -n "$NAMESPACE" get cm "${FULLNAME}-config" -o jsonpath='{.data.LOG_LEVEL}')
[[ "$LOG_LEVEL_NOW" == "INFO" ]] && pass "the aborted upgrade applied no config either" \
                                 || fail "ConfigMap changed to $LOG_LEVEL_NOW despite the abort"

step "A successful upgrade rolls the new config"
helm upgrade "$RELEASE" "$CHART_DIR" -n "$NAMESPACE" -f "$VALUES" "${SC_ARGS[@]}" \
  --set config.logLevel=DEBUG --timeout "$TIMEOUT" --wait >/dev/null
kubectl -n "$NAMESPACE" rollout status "deploy/${FULLNAME}-backend" --timeout=5m >/dev/null
RS_FINAL=$(kubectl -n "$NAMESPACE" get rs -l app.kubernetes.io/component=backend \
  -o jsonpath='{range .items[?(@.spec.replicas>0)]}{.metadata.name}{end}')
[[ "$RS_FINAL" != "$RS_BEFORE" ]] && pass "checksum annotation rolled the pod on a config change" \
                                  || fail "pod was not rolled despite a changed ConfigMap"
LIVE_LOG_LEVEL=$(kubectl -n "$NAMESPACE" exec "deploy/${FULLNAME}-backend" -- sh -c 'echo $LOG_LEVEL')
assert_contains "the new pod sees the new config" "$LIVE_LOG_LEVEL" "DEBUG"

step "Result"
printf '  %d passed, %d failed\n' "$PASS" "$FAIL"
[[ "$FAIL" -eq 0 ]]

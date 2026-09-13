#!/usr/bin/env bash
#
# One-shot minikube bootstrap for KnowledgeDBChat.
#
#   ./deploy/minikube/bootstrap.sh              # start cluster, build images, install
#   SKIP_BUILD=1 ./deploy/minikube/bootstrap.sh # reinstall without rebuilding images
#   SKIP_START=1 ./deploy/minikube/bootstrap.sh # use the running cluster as-is
#
# The chart's images are built from this repo and are not published anywhere, so
# they are built straight into minikube's Docker daemon and referenced with
# pullPolicy=IfNotPresent.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHART_DIR="${REPO_ROOT}/deploy/helm/knowledgedbchat"

PROFILE="${MINIKUBE_PROFILE:-knowledgedbchat}"
NAMESPACE="${NAMESPACE:-knowledgedbchat}"
RELEASE="${RELEASE:-kdbc}"
CPUS="${CPUS:-4}"
MEMORY="${MEMORY:-8192}"
DISK="${DISK:-40g}"
K8S_VERSION="${K8S_VERSION:-v1.31.0}"
IMAGE_TAG="${IMAGE_TAG:-dev}"

# The origin a browser will use to reach the gateway. It is baked into the
# frontend bundle (CRA inlines REACT_APP_* at build time) and into
# config.minioProxyBaseUrl, so both have to agree with how you actually open the
# app. The default assumes the port-forward printed at the end, which works
# identically on every minikube driver - unlike the NodePort, which is not
# routable from the host on the Docker driver.
APP_URL="${APP_URL:-http://localhost:8080}"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

for bin in minikube kubectl helm docker; do
  command -v "$bin" >/dev/null 2>&1 || die "$bin is not installed"
done

# ---------------------------------------------------------------------------
# 1. Cluster
# ---------------------------------------------------------------------------
if [[ "${SKIP_START:-0}" != "1" ]]; then
  log "Starting minikube profile '${PROFILE}' (${CPUS} cpus, ${MEMORY}MB, ${DISK})"
  minikube start \
    --profile="${PROFILE}" \
    --cpus="${CPUS}" \
    --memory="${MEMORY}" \
    --disk-size="${DISK}" \
    --kubernetes-version="${K8S_VERSION}"

  log "Enabling addons"
  minikube addons enable storage-provisioner --profile="${PROFILE}"
  minikube addons enable metrics-server --profile="${PROFILE}"
fi

minikube profile "${PROFILE}" >/dev/null

# ---------------------------------------------------------------------------
# 2. Images (built inside minikube's daemon so no registry is needed)
# ---------------------------------------------------------------------------
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  log "Pointing docker at minikube's daemon"
  eval "$(minikube -p "${PROFILE}" docker-env)"

  log "Building backend image"
  docker build -t "knowledge_db_backend:${IMAGE_TAG}" -f "${REPO_ROOT}/backend/Dockerfile" "${REPO_ROOT}/backend"

  log "Building frontend image for origin ${APP_URL}"
  docker build -t "knowledge_db_frontend:${IMAGE_TAG}" \
    --build-arg "REACT_APP_API_URL=${APP_URL}" \
    --build-arg "REACT_APP_WS_URL=$(printf '%s' "${APP_URL}" | sed -e 's|^http://|ws://|' -e 's|^https://|wss://|')" \
    --build-arg "REACT_APP_VIDEO_STREAM_URL=${APP_URL}/video" \
    -f "${REPO_ROOT}/frontend/Dockerfile" "${REPO_ROOT}/frontend"

  log "Building video-streamer image"
  docker build -t "knowledge_db_video_streamer:${IMAGE_TAG}" -f "${REPO_ROOT}/video-streamer/Dockerfile" "${REPO_ROOT}/video-streamer"

  log "Building mermaid renderer image"
  docker build -t "knowledge_db_mermaid_renderer:${IMAGE_TAG}" -f "${REPO_ROOT}/mermaid-renderer/Dockerfile" "${REPO_ROOT}/mermaid-renderer"

  # The TeX Live image is multi-GB; only built when the LaTeX worker is wanted.
  if [[ "${BUILD_LATEX:-0}" == "1" ]]; then
    log "Building LaTeX worker image"
    docker build -t "knowledge_db_latex_worker:${IMAGE_TAG}" -f "${REPO_ROOT}/backend/Dockerfile.latex-worker" "${REPO_ROOT}/backend"
  fi

  # The transcription worker carries Whisper and the audio stack, and derives
  # FROM the backend image built above -- so it has to come after it, and only
  # when transcription is wanted (celeryTranscription.enabled).
  if [[ "${BUILD_TRANSCRIPTION:-0}" == "1" ]]; then
    log "Building transcription worker image"
    docker build -t "knowledge_db_transcription_worker:${IMAGE_TAG}" \
      --build-arg "BASE_IMAGE=knowledge_db_backend:${IMAGE_TAG}" \
      -f "${REPO_ROOT}/backend/Dockerfile.transcription-worker" "${REPO_ROOT}/backend"
  fi
fi

# ---------------------------------------------------------------------------
# 3. Install
# ---------------------------------------------------------------------------
log "Linting chart"
helm lint "${CHART_DIR}" -f "${CHART_DIR}/values-minikube.yaml"

log "Installing release '${RELEASE}' into namespace '${NAMESPACE}'"
helm upgrade --install "${RELEASE}" "${CHART_DIR}" \
  --namespace "${NAMESPACE}" --create-namespace \
  -f "${CHART_DIR}/values-minikube.yaml" \
  --set image.backend.tag="${IMAGE_TAG}" \
  --set image.frontend.tag="${IMAGE_TAG}" \
  --set image.videoStreamer.tag="${IMAGE_TAG}" \
  --set image.latexWorker.tag="${IMAGE_TAG}" \
  --set celeryLatex.enabled="${BUILD_LATEX:-0}" \
  --set "config.minioProxyBaseUrl=${APP_URL}/minio" \
  --wait --timeout 15m

log "Release is up"
kubectl -n "${NAMESPACE}" get pods -l "app.kubernetes.io/instance=${RELEASE}"

GATEWAY_PORT="${APP_URL##*:}"
cat <<EOF

  Open the app:
    kubectl -n ${NAMESPACE} port-forward svc/${RELEASE}-knowledgedbchat-gateway ${GATEWAY_PORT}:80
    then open ${APP_URL}

  The frontend bundle and MinIO presigned URLs are both built for ${APP_URL}.
  To serve it somewhere else, rerun with APP_URL=... (the frontend image has to
  be rebuilt, so do not combine that with SKIP_BUILD=1).

  API docs:   kubectl -n ${NAMESPACE} port-forward svc/${RELEASE}-knowledgedbchat-backend 8000:8000
              then open http://localhost:8000/docs
  Smoke test: helm test ${RELEASE} -n ${NAMESPACE}
  Logs:       kubectl -n ${NAMESPACE} logs -f deploy/${RELEASE}-knowledgedbchat-backend
  Teardown:   helm uninstall ${RELEASE} -n ${NAMESPACE}  # PVCs are kept
              minikube delete --profile=${PROFILE}

EOF

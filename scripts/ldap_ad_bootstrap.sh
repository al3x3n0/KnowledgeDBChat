#!/usr/bin/env bash
set -euo pipefail

# Bootstrap Active Directory LDAP integration:
# - writes LDAP_* values into backend/.env
# - restarts backend + celery
# - runs /admin/ldap/status and /admin/ldap/import (dry-run by default)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/backend/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing $ENV_FILE (create it from backend/env.example first)"
  exit 1
fi

prompt() {
  local var="$1"
  local msg="$2"
  local secret="${3:-0}"
  local current="${!var:-}"

  if [[ -n "$current" ]]; then
    return 0
  fi

  if [[ "$secret" == "1" ]]; then
    read -r -s -p "$msg: " "$var"
    echo
  else
    read -r -p "$msg: " "$var"
  fi
}

upsert_env() {
  local key="$1"
  local value="$2"
  python3 - "$ENV_FILE" "$key" "$value" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
val = sys.argv[3]

lines = path.read_text(encoding="utf-8").splitlines(True)
pat = re.compile(rf"^\s*{re.escape(key)}\s*=")
out = []
found = False
for ln in lines:
    if pat.match(ln):
        out.append(f"{key}={val}\n")
        found = True
    else:
        out.append(ln)
if not found:
    if out and not out[-1].endswith("\n"):
        out[-1] = out[-1] + "\n"
    out.append(f"{key}={val}\n")
path.write_text("".join(out), encoding="utf-8")
PY
}

LDAP_URI="${LDAP_URI:-}"
LDAP_BASE_DN="${LDAP_BASE_DN:-}"
LDAP_BIND_DN="${LDAP_BIND_DN:-}"
LDAP_BIND_PASSWORD="${LDAP_BIND_PASSWORD:-}"
LDAP_ADMIN_GROUP_DNS="${LDAP_ADMIN_GROUP_DNS:-}"
LDAP_VIEWER_GROUP_DNS="${LDAP_VIEWER_GROUP_DNS:-}"
LDAP_DEFAULT_EMAIL_DOMAIN="${LDAP_DEFAULT_EMAIL_DOMAIN:-}"

ADMIN_USERNAME="${ADMIN_USERNAME:-admin}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-}"

IMPORT_LIMIT="${IMPORT_LIMIT:-200}"
DRY_RUN="${DRY_RUN:-true}"

prompt LDAP_URI "AD LDAP URI (e.g. ldap://dc01.corp.example.com:389)"
prompt LDAP_BASE_DN "AD Base DN (e.g. DC=corp,DC=example,DC=com)"
prompt LDAP_BIND_DN "Bind DN (service account DN)"
prompt LDAP_BIND_PASSWORD "Bind password" 1
prompt ADMIN_PASSWORD "Local admin password (for calling admin import API)" 1

echo "Updating $ENV_FILE ..."
upsert_env LDAP_ENABLED "true"
upsert_env LDAP_URI "$LDAP_URI"
upsert_env LDAP_START_TLS "${LDAP_START_TLS:-false}"
upsert_env LDAP_INSECURE_SKIP_TLS_VERIFY "${LDAP_INSECURE_SKIP_TLS_VERIFY:-false}"
upsert_env LDAP_CONNECT_TIMEOUT_SECONDS "${LDAP_CONNECT_TIMEOUT_SECONDS:-8}"

upsert_env LDAP_BIND_DN "$LDAP_BIND_DN"
upsert_env LDAP_BIND_PASSWORD "$LDAP_BIND_PASSWORD"
upsert_env LDAP_BASE_DN "$LDAP_BASE_DN"
upsert_env LDAP_USER_DN_TEMPLATE ""
upsert_env LDAP_USER_SEARCH_FILTER "${LDAP_USER_SEARCH_FILTER:-(&(objectClass=user)(!(objectClass=computer))(|(sAMAccountName={username})(userPrincipalName={username})))}"
upsert_env LDAP_IMPORT_FILTER "${LDAP_IMPORT_FILTER:-(&(objectClass=user)(!(objectClass=computer)))}"

upsert_env LDAP_USERNAME_ATTRIBUTE "${LDAP_USERNAME_ATTRIBUTE:-sAMAccountName}"
upsert_env LDAP_EMAIL_ATTRIBUTE "${LDAP_EMAIL_ATTRIBUTE:-mail}"
upsert_env LDAP_FULL_NAME_ATTRIBUTE "${LDAP_FULL_NAME_ATTRIBUTE:-displayName}"
upsert_env LDAP_GROUPS_ATTRIBUTE "${LDAP_GROUPS_ATTRIBUTE:-memberOf}"
upsert_env LDAP_DEFAULT_EMAIL_DOMAIN "${LDAP_DEFAULT_EMAIL_DOMAIN}"
upsert_env LDAP_USER_ATTRIBUTES "${LDAP_USER_ATTRIBUTES:-sAMAccountName,userPrincipalName,mail,displayName,cn,memberOf}"

upsert_env LDAP_ADMIN_GROUP_DNS "$LDAP_ADMIN_GROUP_DNS"
upsert_env LDAP_VIEWER_GROUP_DNS "$LDAP_VIEWER_GROUP_DNS"
upsert_env LDAP_SYNC_ON_LOGIN "${LDAP_SYNC_ON_LOGIN:-true}"
upsert_env LDAP_CREATE_USER_ON_LOGIN "${LDAP_CREATE_USER_ON_LOGIN:-true}"

echo "Restarting services..."
cd "$ROOT_DIR"
docker compose restart backend celery

echo "Checking LDAP status..."
API="http://localhost:8000/api/v1"
TOKEN="$(curl -s -X POST "$API/auth/login" -H "Content-Type: application/json" -d "{\"username\":\"$ADMIN_USERNAME\",\"password\":\"$ADMIN_PASSWORD\"}" | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')"
curl -s "$API/admin/ldap/status" -H "Authorization: Bearer $TOKEN" | python3 -m json.tool

echo "Running LDAP import (dry_run=$DRY_RUN, limit=$IMPORT_LIMIT)..."
curl -s -X POST "$API/admin/ldap/import" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"dry_run\":$DRY_RUN,\"limit\":$IMPORT_LIMIT}" | python3 -m json.tool

echo "Done."

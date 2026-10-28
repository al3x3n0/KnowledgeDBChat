#!/usr/bin/env bash
# Start a disposable slapd for tests/test_ldap_real_server.py.
#
#   backend/scripts/ldap_test_server.sh <cert-dir>
#
# Makes a CA and a certificate for "localhost" in <cert-dir> (the image's own
# CA expired years ago, and the client rightly refuses it), starts
# osixia/openldap with LDAPS on 127.0.0.1:3636 and plain/StartTLS on :3389,
# and waits until the administrator can bind. Then:
#
#   TEST_LDAP_URI=ldaps://localhost:3636 TEST_LDAP_CA_FILE=<cert-dir>/ca.crt \
#     pytest tests/test_ldap_real_server.py
#
# and ldap://localhost:3389 for the same tests over StartTLS. Remove it with
# `docker rm -f kdbc-ldap`. CI runs exactly this (the ldap-real-server job).
set -euo pipefail

dir="${1:?usage: $0 <cert-dir>}"
name="${LDAP_TEST_CONTAINER:-kdbc-ldap}"
image="osixia/openldap:1.5.0"
mkdir -p "$dir"
dir="$(cd "$dir" && pwd)"

openssl req -x509 -newkey rsa:2048 -nodes -keyout "$dir/ca.key" \
  -out "$dir/ca.crt" -days 30 -subj /CN=kdbc-test-ca 2>/dev/null
openssl req -newkey rsa:2048 -nodes -keyout "$dir/ldap.key" \
  -out "$dir/ldap.csr" -subj /CN=localhost 2>/dev/null
printf 'subjectAltName=DNS:localhost\nextendedKeyUsage=serverAuth\n' >"$dir/ext"
openssl x509 -req -in "$dir/ldap.csr" -CA "$dir/ca.crt" -CAkey "$dir/ca.key" \
  -CAcreateserial -out "$dir/ldap.crt" -days 30 -extfile "$dir/ext" 2>/dev/null
# The image reads the key as its own user.
chmod 644 "$dir/ldap.key"

docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" --hostname localhost \
  -p 127.0.0.1:3389:389 -p 127.0.0.1:3636:636 \
  -e LDAP_DOMAIN=example.com -e LDAP_ADMIN_PASSWORD=admin-secret \
  -e LDAP_TLS_VERIFY_CLIENT=never -e LDAP_TLS_CRT_FILENAME=ldap.crt \
  -e LDAP_TLS_KEY_FILENAME=ldap.key -e LDAP_TLS_CA_CRT_FILENAME=ca.crt \
  -v "$dir":/container/service/slapd/assets/certs "$image" >/dev/null

# The image restarts slapd once while it configures itself, so a single
# successful bind is not "ready": wait for two in a row, over both listeners
# (plain LDAP answered while LDAPS still dropped the handshake).
bind() {
  docker exec -e LDAPTLS_CACERT=/container/service/slapd/assets/certs/ca.crt \
    "$name" ldapwhoami -x -H "$1" \
    -D cn=admin,dc=example,dc=com -w admin-secret >/dev/null 2>&1
}
ok=0
for _ in $(seq 1 90); do
  if bind ldap://localhost && bind ldaps://localhost; then
    ok=$((ok + 1))
    [ "$ok" -ge 2 ] && { echo "slapd is ready ($name)"; exit 0; }
  else
    ok=0
  fi
  sleep 2
done
echo "slapd did not become ready; its log:" >&2
docker logs "$name" >&2 || true
exit 1

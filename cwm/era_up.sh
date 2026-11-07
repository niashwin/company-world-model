#!/bin/sh
# Start the seven Era by Eon simulators for one company (same seed => the same company in every system).
# Usage:  sh era_up.sh [day1|day2]
# Needs Docker. Images are pulled from Docker Hub (erabyeon/*). Optional environment:
#   SIMCORE_SEED (default 42), SIMCORE_TODAY (default 2026-10-31),
#   ERA_CA_BUNDLE=/path/to/ca.crt  if your network intercepts TLS (mounted into the containers).
STATE=${1:-day1}
SEED=${SIMCORE_SEED:-42}
TODAY=${SIMCORE_TODAY:-2026-10-31}
EXTRA=""
if [ -n "$ERA_CA_BUNDLE" ]; then
  EXTRA="-v $ERA_CA_BUNDLE:/etc/ssl/certs/proxy-ca.crt:ro -e SSL_CERT_FILE=/etc/ssl/certs/proxy-ca.crt -e UV_NATIVE_TLS=1 -e UV_SYSTEM_CERTS=1"
fi
if [ -n "$HTTPS_PROXY" ]; then
  EXTRA="$EXTRA -e HTTPS_PROXY=$HTTPS_PROXY -e https_proxy=$HTTPS_PROXY -e NO_PROXY=$NO_PROXY"
fi
start() { # name image port
  docker rm -f "era-$1" >/dev/null 2>&1
  # shellcheck disable=SC2086
  docker run -d --name "era-$1" --network host $EXTRA \
    -e SIMCORE_SEED="$SEED" -e SIMCORE_TODAY="$TODAY" -e SIMCORE_FIDELITY=high -e SIMCORE_STATE="$STATE" \
    "erabyeon/$2:latest" --host 127.0.0.1 --port "$3" >/dev/null && echo "started $1 on :$3"
}
start slack slack 8080
start salesforce salesforce 8081
start zendesk zendesk 8082
start gong gong 8083
start gmail gmail 8084
start calendar google-calendar 8085
start jira jira-confluence 8087

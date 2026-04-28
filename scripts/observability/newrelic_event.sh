#!/usr/bin/env bash
set -euo pipefail

repo="corinth-canal"
command_name="${1:-manual}"
latency_ms="${2:-0}"
success="${3:-true}"
error_category="${4:-none}"
environment="${SENTRY_ENVIRONMENT:-local}"
git_sha="${AGENTOS_GIT_SHA:-$(git rev-parse --short HEAD 2>/dev/null || printf 'unknown')}"
run_id="${AGENTOS_RUN_ID:-${repo}-${git_sha}}"
release="${repo}@${git_sha}"
entity_search="${NEW_RELIC_ENTITY_SEARCH_CORINTH_CANAL:-}"
actor="${NEW_RELIC_USER:-agentos}"

if [[ -z "${NEW_RELIC_ACCOUNT_ID:-}" ]]; then
  printf 'NEW_RELIC_ACCOUNT_ID is required\n' >&2
  exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
  printf 'jq is required to build the New Relic event payload safely\n' >&2
  exit 2
fi

case "${success}" in
  true|false) ;;
  *) success=false ;;
esac

if ! [[ "${latency_ms}" =~ ^[0-9]+$ ]]; then
  latency_ms=0
fi

event="$(
  jq -n \
    --arg eventType "AgentOsRepoRun" \
    --arg repo "${repo}" \
    --arg run_id "${run_id}" \
    --arg git_sha "${git_sha}" \
    --arg command "${command_name}" \
    --arg error_category "${error_category}" \
    --arg environment "${environment}" \
    --argjson latency_ms "${latency_ms}" \
    --argjson success "${success}" \
    '{eventType: $eventType, repo: $repo, run_id: $run_id, git_sha: $git_sha, command: $command, latency_ms: $latency_ms, success: $success, error_category: $error_category, environment: $environment}'
)"

newrelic events post --event "${event}"

if [[ -n "${entity_search}" ]]; then
  newrelic changeTracking create \
    --entitySearch "${entity_search}" \
    --category Deployment \
    --type Basic \
    --version "${release}" \
    --commit "${git_sha}" \
    --user "${actor}"
fi

#!/usr/bin/env bash
set -euo pipefail

repo="corinth-canal"
org="${SENTRY_ORG:-}"
project="${SENTRY_PROJECT_CORINTH_CANAL:-}"
environment="${SENTRY_ENVIRONMENT:-local}"
git_sha="${AGENTOS_GIT_SHA:-$(git rev-parse --short HEAD 2>/dev/null || printf 'unknown')}"
release="${repo}@${git_sha}"

if [[ -z "${org}" ]]; then
  printf 'SENTRY_ORG is required\n' >&2
  exit 2
fi

if [[ -z "${project}" ]]; then
  printf 'SENTRY_PROJECT_CORINTH_CANAL is required\n' >&2
  exit 2
fi

if ! sentry-cli releases --org "${org}" --project "${project}" info "${release}" >/dev/null 2>&1; then
  sentry-cli releases --org "${org}" --project "${project}" new "${release}"
fi

sentry-cli releases --org "${org}" --project "${project}" set-commits "${release}" --auto --ignore-missing
sentry-cli releases --org "${org}" --project "${project}" finalize "${release}"
sentry-cli deploys --org "${org}" --project "${project}" new --release "${release}" -e "${environment}"

printf '%s\n' "${release}"

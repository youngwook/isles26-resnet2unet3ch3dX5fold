#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# DOCKER_IMAGE_TAG="example_algorithm_preliminary-docker-evaluation-sanity-check"
DOCKER_IMAGE_TAG="docker-evaluation-sanity-checks"

# docker build \
#   --platform=linux/amd64 \
#   --tag "$DOCKER_IMAGE_TAG"  \
#   ${DOCKER_QUIET_BUILD:+--quiet} \
#   "$SCRIPT_DIR" 2>&1
docker build \
  --no-cache \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG"  \
  ${DOCKER_QUIET_BUILD:+--quiet} \
  "$SCRIPT_DIR" 2>&1

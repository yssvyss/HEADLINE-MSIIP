#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="example_algorithm_sanity-check"

# The build context will be set to CHALLENGE_ROOT in `docker build` below.
# Depending on your project structure, you may need to adjust this.
# In particular, whatever files/directories you COPY in the Dockerfile must be reachable
# from the build context. Paths in the Dockerfile are relative to the build context.
CHALLENGE_ROOT="$SCRIPT_DIR"  # or e.g., "$SCRIPT_DIR/../../.." for several levels up.
echo "Using inference backend: swift infer --infer_backend pt"

docker build \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG"  \
  --file "$SCRIPT_DIR/Dockerfile" \
  ${DOCKER_QUIET_BUILD:+--quiet} \
  "$CHALLENGE_ROOT" 2>&1

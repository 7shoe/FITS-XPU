#!/usr/bin/env bash
# Alias for the documented model-explicit pretraining command.
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_pretrain.sh" "$@"

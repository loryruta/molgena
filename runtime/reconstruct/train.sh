#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

DATASET_DIR="$SCRIPT_DIR/.."
SRC_DIR="$SCRIPT_DIR/../../src"

cd "$SRC_DIR"

export PYTHONPATH=$SRC_DIR
export DATASET_DIR=$DATASET_DIR

python ./train/train_reconstruct.py \
  --train-dir "$SCRIPT_DIR/" \
  --config "$SCRIPT_DIR/configs/molgena-1b.json"

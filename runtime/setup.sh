#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

cd "$SCRIPT_DIR/../src"

export DATASET_DIR=$SCRIPT_DIR

# Download the dataset
python gen_dataset.py generate || exit

# Split the dataset into training/validation/test set
python gen_dataset.py split --training-frac 0.8 --validation-frac 0.1 --seed 23 || exit

# Construct the motif vocabulary from the training set
python gen_motif_vocab.py || exit

# Filter the dataset from molecules that can't be constructed with the motif vocabulary, or for which motif graph
# identity doesn't work (quick-fix)
python gen_dataset.py filter || exit

# Generate motif graphs for training/validation/test sets; output .pkl files
python gen_motif_graphs.py || exit

DATASET_SIZE=$(du -sh "$DATASET_DIR" | awk '{print $1}')
echo "Dataset generation done! Size: $DATASET_SIZE"

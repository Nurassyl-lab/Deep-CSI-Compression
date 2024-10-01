#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Function to run a Python command
run_python_command() {
    local command=$1
    printf "Running command: ${BLUE}$command${NC}\n"
    eval "$command"
    if [ $? -eq 0 ]; then
        printf "${GREEN}Completed successfully!${NC}\n\n"
    else
        printf "${RED}Command failed. Exiting.${NC}\n"
        exit 1
    fi
}

# Train
batch_prep="python autoencoder_train_test.py --model CNN_het10_class1 --dataset SNR_biased_10_1_dataset.csv --process train --epochs 100 --batch_size 32 --lr 0.001"
run_python_command "$batch_prep"

batch_prep="python autoencoder_train_test.py --model CNN_het10_class2 --dataset SNR_biased_10_2_dataset.csv --process train --epochs 100 --batch_size 32 --lr 0.001"
run_python_command "$batch_prep"

batch_prep="python autoencoder_train_test.py --model CNN_het10_class3 --dataset SNR_biased_10_3_dataset.csv --process train --epochs 100 --batch_size 32 --lr 0.001"
run_python_command "$batch_prep"

# Test
batch_prep="python autoencoder_train_test.py --model CNN_het10_class1 --dataset SNR_biased_10_1_dataset.csv --process test --epochs 100 --batch_size 32 --lr 0.001"
run_python_command "$batch_prep"

batch_prep="python autoencoder_train_test.py --model CNN_het10_class2 --dataset SNR_biased_10_2_dataset.csv --process test --epochs 100 --batch_size 32 --lr 0.001"
run_python_command "$batch_prep"

batch_prep="python autoencoder_train_test.py --model CNN_het10_class3 --dataset SNR_biased_10_3_dataset.csv --process test --epochs 100 --batch_size 32 --lr 0.001"
run_python_command "$batch_prep"

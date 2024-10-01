#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m'


# Parse command-line options
while getopts "d:m:n:p:l:" opt; do
    case ${opt} in
        d )
            dataset=$OPTARG
            ;;
        m )
            model=$OPTARG
            ;;
        n )
            epochs=$OPTARG
            ;;
        p )
            process=$OPTARG
            ;;
        l )
            limit=$OPTARG
            ;;
    esac
done

# -d is the dataset to be used for the quality scoring process
# -i is the name of the folder where the batch input files will be stored
# -o is the name of the folder where the batch output files will be stored
# -m is the pre-processing model to be used
# -h is the hop value for the quality scoring process


# Check if the folder exists, if it does, add v+# to the folder name,where # is the next available number
# if it does not exist, create the folder

# if l get None value set it to 0
if [ -z "$limit" ]; then
    limit=0
fi

# Function to print a header
print_header() {
    # Length of the header content
    local header_length=${#1}
    local total_length=50
    local padding_length=$(( (total_length - header_length) / 2 ))

    # Print top border
    printf "\n${YELLOW}┌"
    printf '─%.0s' $(seq 1 $total_length)
    printf "┐\n"

    # Print the header content with padding
    printf "│"
    printf '%*s' $(( padding_length + header_length )) "$1"
    printf '%*s' $(( total_length - padding_length - header_length )) ""
    printf "│\n"

    # Print bottom border
    printf "└"
    printf '─%.0s' $(seq 1 $total_length)
    printf "┘${NC}\n"
}


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


# Display the information about this run
# printf "\nStarting the quality scoring process on the dataset ${GREEN}$dataset${NC} for a ${GREEN}$hop${NC} hop task\n"
# printf "Pre-processing for model: ${GREEN}$model${NC}\n"
# printf "Batch input file(s) will be stored in: ${GREEN}./data/batch_input/$batch_input/${NC} folder\n"
# printf "Batch output file(s) will be stored in: ${GREEN}./data/batch_output/$batch_output/${NC} folder\n\n"


# Train
print_header "Training the model"
batch_prep="python autoencoder_train_test.py --model $model --dataset $dataset --process $process --epochs $epochs --batch_size 32 --lr 0.001 --limit $limit"
run_python_command "$batch_prep"

# Quality Scoring
# print_header "Starting the quality scoring process"
# batch_proc="python path_quality_scorer_batch.py --input_folder $batch_input --output_folder $batch_output --model $model --hop $hop --monitor True"
# run_python_command "$batch_proc"

# # Post-processing
# print_header "Post-processing the batch output files"
# batch_post="python batch_output_processing.py --input_dataset $dataset --input_folder $batch_output --model $model"
# run_python_command "$batch_post"

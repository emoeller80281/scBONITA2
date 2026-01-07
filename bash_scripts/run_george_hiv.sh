#!/bin/bash -l

CONDA_ENV_NAME="scBonita"
# -------------- User Input --------------

# IMPORTANT!!! MAKE SURE THAT THERE ARE NO SPACES IN FILE NAMES

# =============================================
# SELECT WHICH PROCESSES TO RUN
# =============================================
# Which parts do you want to run? Set True to run or False to skip
    # Rule determination must be run prior to importance score, importance score must be run prior to relative abundance
RUN_RULE_DETERMINATION=false
RUN_IMPORTANCE_SCORE=false
RUN_RELATIVE_ABUNDANCE=false
RUN_ATTRACTOR_ANALYSIS=true

# General Arguments (Required for all steps)
# HIV_dataset_normalized_integrated_counts
DATA_FILE="./input/george_HIV_data.csv"
DATASET_NAME="george_hiv" # Enter the name of your dataset
DATAFILE_SEP="," # Enter the character that the values in your dataset are split by

# "04064" "04630" "04620" "04666" "04060" "04210" "04150" "04010" "04621"
KEGG_PATHWAYS=("05417") # Enter KEGG pathway codes or leave blank to find all pathways with overlapping genes. Separate like: ("04670" "05171")
CUSTOM_PATHWAYS=() #("modified_network.graphml") #Put custom networks in the input folder
BINARIZE_THRESHOLD=0.01 # Data points with values above this number will be set to 1, lower set to 0
MINIMUM_OVERLAP=1 # Specifies how many genes you want to ensure overlap with the genes in the KEGG pathways. Default is 25
ORGANISM_CODE="hsa" # Organism code in front of KEGG pathway numbers

# Relative Abundance arguments
METADATA_FILE="input/george_HIV_metadata.txt"
METADATA_SEP=" "
HEADER="n" # Does the metadata file contain a header before the entries start?
OVERWRITE="n" # Do you want to overwrite the files generated for each of your different experimental groups?
CELL_NAME_COL=1 # What column contains the cell names (first column = 0)
GROUP_INDICES=(2)

# Specify the control groups and experimental groups that you want to compare
    # 1st entry in control is compared to 1st entry in experimental, 2nd entry compared to 2nd entry, etc.
CONTROL_GROUPS=("Healthy")
EXPERIMENTAL_GROUPS=("HIV")

# Attractor Analysis Arguments
NUM_CELLS_PER_CHUNK=250 # The number of cells in each chunk to summarize the cluster trajectories
NUM_CELLS_TO_ANALYZE=3621 # The total number of cells to analyze

LOG_DIR="LOGS/${DATASET_NAME}/"
mkdir -p "${LOG_DIR}"

# Set output and error files dynamically
exec > "${LOG_DIR}/main_pipeline.log" 2> "${LOG_DIR}/main_pipeline.err"

# -------------- End of user input, shouldn't have to change anything below here --------------

# =============================================
# FUNCTIONS
# =============================================

# -------------- VALIDATION FUNCTIONS ----------------------
# Check if there are any jobs with the same name running before starting
check_for_running_jobs() {
    echo "[INFO] Checking for running jobs with the same name..."
    if [ -z "${SLURM_JOB_NAME:-}" ]; then
        echo "    Not running in a SLURM environment, not checking for running tasks"
        return 0
    fi

    # Use the SLURM job name for comparison
    JOB_NAME="${SLURM_JOB_NAME:-scBONITA2}"  # Dynamically retrieve the job name from SLURM

    # Check for running jobs with the same name, excluding the current job
    RUNNING_COUNT=$(squeue --name="$JOB_NAME" --noheader | wc -l)

    # If other jobs with the same name are running, exit
    if [ "$RUNNING_COUNT" -gt 1 ]; then
        echo "[WARNING] A job with the name '"$JOB_NAME"' is already running:"
        echo "    Exiting to avoid conflicts."
        exit 1
    
    # If no other jobs are running, pass
    else
        echo "    No other jobs with the name '"$JOB_NAME"'"
    fi
}

# Function to check if at least one process is selected
check_pipeline_steps() {
    if ! $RUN_RULE_DETERMINATION \
    && ! $RUN_IMPORTANCE_SCORE \
    && ! $RUN_RELATIVE_ABUNDANCE \
    && ! $RUN_ATTRACTOR_ANALYSIS; then \
        echo "Error: At least one process must be enabled to run the pipeline."
        exit 1
    fi
}

# Determine the number of cpus allocated to this job
determine_num_cpus() {
    if [ -z "${SLURM_CPUS_PER_TASK:-}" ]; then
        if command -v nproc &> /dev/null; then
            TOTAL_CPUS=$(nproc --all)
            case $TOTAL_CPUS in
                [1-15]) IGNORED_CPUS=1 ;;  # Reserve 1 CPU for <=15 cores
                [16-31]) IGNORED_CPUS=2 ;; # Reserve 2 CPUs for <=31 cores
                *) IGNORED_CPUS=4 ;;       # Reserve 4 CPUs for >=32 cores
            esac
            NUM_CPU=$((TOTAL_CPUS - IGNORED_CPUS))
            echo "[INFO] Running locally. Detected $TOTAL_CPUS CPUs, reserving $IGNORED_CPUS for system tasks. Using $NUM_CPU CPUs."
        else
            NUM_CPU=1  # Fallback
            echo "[INFO] Running locally. Unable to detect CPUs, defaulting to $NUM_CPU CPU."
        fi
    else
        NUM_CPU=${SLURM_CPUS_PER_TASK}
        echo "[INFO] Running on SLURM. Number of CPUs allocated: ${NUM_CPU}"
    fi
}

# Function to activate Conda environment
activate_conda_env() {
    CONDA_BASE=$(conda info --base)
    if [ -z "$CONDA_BASE" ]; then
        echo "Error: Conda base could not be determined. Is Conda installed and in your PATH?"
        exit 1
    fi

    source "$CONDA_BASE/bin/activate"
    if ! conda env list | grep -q "^$CONDA_ENV_NAME "; then
        echo "Error: Conda environment '$CONDA_ENV_NAME' does not exist."
        exit 1
    fi

    conda activate "$CONDA_ENV_NAME" || { echo "Error: Failed to activate Conda environment '$CONDA_ENV_NAME'."; exit 1; }
    echo "Activated Conda environment: $CONDA_ENV_NAME"

    CONDA_ENVIRONMENT_PYTHON="$(command -v python)"
    export CONDA_ENVIRONMENT_PYTHON
    echo "[INFO] Using python: $CONDA_ENVIRONMENT_PYTHON"


}

run_rule_determination() {
    echo "Running Rule Determination..."


    if [ ${#KEGG_PATHWAYS[@]} -gt 0 ]; then
        echo "Running with KEGG pathways"

        # Using a list of KEGG pathways:
        KEGG_PATHWAYS_ARGS="${KEGG_PATHWAYS[@]}"

        /usr/bin/time -v \
        $CONDA_ENVIRONMENT_PYTHON code/pipeline_class.py \
            --data_file "$DATA_FILE" \
            --dataset_name "$DATASET_NAME" \
            --datafile_sep "$DATAFILE_SEP" \
            --list_of_kegg_pathways $KEGG_PATHWAYS_ARGS \
            --binarize_threshold $BINARIZE_THRESHOLD \
            --organism $ORGANISM_CODE \
            --minimum_overlap $MINIMUM_OVERLAP
    else
        echo "No KEGG pathways specified, finding kegg pathways with overlapping genes..."
        /usr/bin/time -v \
        $CONDA_ENVIRONMENT_PYTHON code/pipeline_class.py \
        --data_file "$DATA_FILE" \
        --dataset_name "$DATASET_NAME" \
        --datafile_sep "$DATAFILE_SEP" \
        --get_kegg_pathways True \
        --binarize_threshold $BINARIZE_THRESHOLD \
        --organism $ORGANISM_CODE \
        --minimum_overlap $MINIMUM_OVERLAP
    fi

    # Using a custom network saved to the scBONITA directory:

    # Check and execute for Custom Pathways if the array is not empty
    if [ ${#CUSTOM_PATHWAYS[@]} -gt 0 ]; then
        echo "Running with Custom Pathways..."
        
        CUSTOM_PATHWAYS_ARGS=""
        for pathway in "${CUSTOM_PATHWAYS[@]}"; do
            CUSTOM_PATHWAYS_ARGS+="--network_files $pathway "
        done

        $CONDA_ENVIRONMENT_PYTHON pipeline_class.py \
        --data_file "$DATA_FILE" \
        --dataset_name "$DATASET_NAME" \
        --datafile_sep "$DATAFILE_SEP" \
        $CUSTOM_PATHWAYS_ARGS \
        --binarize_threshold $BINARIZE_THRESHOLD \
        --get_kegg_pathways "False" \
        --minimum_overlap $MINIMUM_OVERLAP
    else
        echo "No Custom Pathways specified, skipping this part..."
    fi
} 2> "$LOG_DIR/rule_determination.log"

run_importance_score() {
    echo "Running Importance Score Calculation..."

    if [ ${#KEGG_PATHWAYS[@]} -gt 0 ]; then
        echo "Running with KEGG pathways"
        # Using a list of KEGG pathways:
        KEGG_PATHWAYS_ARGS="${KEGG_PATHWAYS[@]}"

        /usr/bin/time -v \
        $CONDA_ENVIRONMENT_PYTHON code/importance_scores.py \
            --dataset_name "$DATASET_NAME" \
            --list_of_kegg_pathways $KEGG_PATHWAYS_ARGS
    else
        echo "No KEGG Pathways specified"
    fi
} 2> "$LOG_DIR/importance_score.log"

run_relative_abundance() {
    echo "Running Relative Abundance Calculations..."

    GROUP_INDICES_ARGS="${GROUP_INDICES[@]}"

    # Check that both arrays have the same length
    if [ ${#CONTROL_GROUPS[@]} -ne ${#EXPERIMENTAL_GROUPS[@]} ]; then
        echo "Control and Experimental groups arrays do not match in length!"
        exit 1
    fi

    if [ ${#KEGG_PATHWAYS[@]} -gt 0 ]; then
        echo "Running with KEGG pathways"
        # Using a list of KEGG pathways:
        KEGG_PATHWAYS_ARGS="${KEGG_PATHWAYS[@]}"
    else
        echo "No KEGG Pathways specified"
    fi

    # Loop through the control and experimental groups
    for (( i=0; i<${#CONTROL_GROUPS[@]}; i++ )); do

        # Extract the current pair of control and experimental group
        CONTROL_GROUP=${CONTROL_GROUPS[$i]}
        EXPERIMENTAL_GROUP=${EXPERIMENTAL_GROUPS[$i]}

        # Execute the command with the current pair of control and experimental group
        /usr/bin/time -v \
        $CONDA_ENVIRONMENT_PYTHON code/relative_abundance.py \
            --dataset_name "$DATASET_NAME" \
            --dataset_file "$DATA_FILE" \
            --metadata_file "$METADATA_FILE" \
            --metadata_sep "$METADATA_SEP" \
            --dataset_sep "$DATAFILE_SEP" \
            --control_group "$CONTROL_GROUP" \
            --experimental_group "$EXPERIMENTAL_GROUP" \
            --cell_name_index $CELL_NAME_COL \
            --group_indices $GROUP_INDICES_ARGS \
            --header "$HEADER" \
            --overwrite "$OVERWRITE" \
            --organism "$ORGANISM_CODE" \
            --list_of_kegg_pathways $KEGG_PATHWAYS_ARGS
        done

} 2> "$LOG_DIR/relative_abundance.log"

run_attractor_analysis() {
    echo "Running Attractor Analysis..."

    /usr/bin/time -v \
    $CONDA_ENVIRONMENT_PYTHON code/attractor_analysis.py \
        --dataset_name "$DATASET_NAME" \
        --num_cells_per_chunk $NUM_CELLS_PER_CHUNK \
        --num_cells_to_analyze $NUM_CELLS_TO_ANALYZE
} 2> "$LOG_DIR/attractor_analysis.log"

# =============================================
# MAIN PIPELINE
# =============================================

# Run the functions
check_for_running_jobs
check_pipeline_steps
determine_num_cpus
activate_conda_env

set -euo pipefail

if [ "$RUN_RULE_DETERMINATION" = true ]; then run_rule_determination; fi
if [ "$RUN_IMPORTANCE_SCORE" = true ]; then run_importance_score; fi
if [ "$RUN_RELATIVE_ABUNDANCE" = true ]; then run_relative_abundance; fi
if [ "$RUN_ATTRACTOR_ANALYSIS" = true ]; then run_attractor_analysis; fi

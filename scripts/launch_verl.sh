#!/bin/bash

# -----------------------------------
# Script: launch_verl.sh
# Description: Launches a SLURM job for training using Ray with VERL dependencies.
# -----------------------------------

# If run from an interactive shell, make sure that SLURM environment variables are not set
EDF_EXPANDED="$SLURM_EDF_EXPANDED"
unset ${!SLURM_*}
set -e

# Global, constants
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
BLUE='\033[1;34m'
RED='\033[1;31m'
NC='\033[0m' # No Color
PURPOSE_STRING="Launch a SLURM job for training using Ray with VERL dependencies."
USAGE_STRING=$(cat <<EOF
Usage: $0 --config <config_name> [--num-nodes <num_nodes>]
         [--report-dir <report_dir>] [--timeout <timeout>]
         [--no-confirm] [--help]
Options:
  --config <config_name>    (Required) Path to the configuration file.
  --environment             (Optional) Path to the environment, default to \`~/.edf/multimodal.toml\`.
  --experiment-name <name>  (Optional) Name of the experiment. Default is derived from the date and time.
  --num-nodes <num_nodes>   (Optional) Number of nodes to use. If not provided, the user will be prompted.
  --report-dir <report_dir> (Optional) Directory to save reports. Default is './reports/verl'.
  --venv-dir <venv_dir>     (Optional) Directory to store the virtual environment to. Default to './.venv'.
  --timeout <timeout>       (Optional) Job timeout in HH:MM:SS format. Default is '11:59:00'.
  --no-confirm              If set, skips the confirmation prompt before job submission.
  --recreate-venv           If set, recreate the venv and rerun the install script.
  --help                    (Optional) Display this help message and exit.
EOF
)

# Detect login node (hostname is of format `clariden-lnXXX`)
HOSTNAME=$(hostname)
IS_LOGIN_NODE=false
CURRENT_EDF_IMAGE=""
if [[ "$HOSTNAME" == *-ln* ]]; then
    IS_LOGIN_NODE=true
    echo -e "${BLUE}Detected, running on login node, cannot create virtual environment here.${NC}"
else
    echo -e "${BLUE}Running on compute node $HOSTNAME.${NC}"

    # We expect the SLURM_EDF_EXPANDED to contains a key image = "..."
    if [ ! -z "$EDF_EXPANDED" ]; then
        CURRENT_EDF_IMAGE=$(echo "$EDF_EXPANDED" | grep -oP '^\s*image\s*=\s*"\K[^"]+')
    else
        CURRENT_EDF_IMAGE=""
    fi
fi

# Default values
CONFIG_NAME=""
NUM_NODES=""
REPORT_DIR=""
NO_CONFIRM=""
RECREATE_VENV=""
VENV_DIR="./.venv"
ENVIRONMENT="$HOME/.edf/multimodal.toml"
TIMEOUT="11:59:00"
EXPERIMENT_NAME="verl-$(date +%Y%m%d-%H%M%S)"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    key="$1"
    case $key in
        --config)
            CONFIG_NAME="$2"
            shift # past argument
            shift # past value
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift # past argument
            shift # past value
            ;;
        --report-dir)
            REPORT_DIR="$2"
            shift # past argument
            shift # past value
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift # past argument
            shift # past value
            ;;
        --num-nodes)
            NUM_NODES="$2"
            if ! [[ "$NUM_NODES" =~ ^[0-9]+$ ]]; then
                echo "Error: --num-nodes must be an integer."
                echo "$USAGE_STRING"
                exit 1
            fi
            shift # past argument
            shift # past value
            ;;
        --venv-dir)
            VENV_DIR="$2"
            shift # past argument
            shift # past value
            ;;
        --timeout)
            TIMEOUT="$2"
            shift # past argument
            shift # past value
            ;;
        --no-confirm)
            NO_CONFIRM="1"
            shift # past argument
            ;;
        --recreate-venv)
            RECREATE_VENV="1"
            shift # past argument
            ;;
        --help)
            echo $PURPOSE_STRING
            echo "$USAGE_STRING"
            exit 0
            ;;
        *)
        echo "Unknown parameter passed: $1"
        echo "$USAGE_STRING"
        exit 1
        ;;
    esac
done

# Find the image corresponding with the current environemtn file
ENVIRONMENT_EDF_IMAGE=$(grep -oP '^\s*image\s*=\s*"\K[^"]+' "$ENVIRONMENT")
echo -e "${GREEN}Using environment file: $ENVIRONMENT which specifies image: $ENVIRONMENT_EDF_IMAGE${NC}"

# Check the config name is provided
if [ -z "$CONFIG_NAME" ]; then
    echo "Error: --config <config_name> is required."
    echo "$USAGE_STRING"
    exit 1
fi
CONFIG_NAME="$(realpath "$CONFIG_NAME")"

# If the num-nodes is not provided, ask the user
if [ -z "$NUM_NODES" ]; then
    read -p "Enter the number of nodes to use: " NUM_NODES
    if ! [[ "$NUM_NODES" =~ ^[0-9]+$ ]]; then
        echo "Error: --num-nodes must be an integer."
        exit 1
    fi
fi

# Check that the venv dir exists, if not create and install it
if [ ! -d "$VENV_DIR" ] || [ ! -z "$RECREATE_VENV" ]; then
    # Ensure not on the login node
    if [ "$IS_LOGIN_NODE" = true ]; then
        echo -e "${RED}Error: Cannot create or recreate virtual environment on login node.${NC}"
        exit 1
    fi

    # Check current environment file matches the expected one
    if [ -z "$CURRENT_EDF_IMAGE" ]; then
        echo "${YELLOW}WARNING: Could not determine the current EDF environment image. Proceeding anyway.${NC}"
    else
        if [ "$CURRENT_EDF_IMAGE" != "$ENVIRONMENT_EDF_IMAGE" ]; then
            echo -e "${RED}Error: The current EDF environment file ($CURRENT_EDF_IMAGE) does not match the specified one ($ENVIRONMENT).${NC}"
            exit 1
        fi
    fi

    echo "Generating virtual environment for running the script at $VENV_DIR"
    echo -e "${YELLOW}WARNING: This will (re)create the virtual environment and install dependencies, which may take some time."
    echo -e "Make sure that you are running this script with the barebones environment, you SHOULD NOT DO ANY PIP INSTALLS PRIOR TO THIS STEP."
    echo -e "This is because the virtual environment is created with system-site-packages, if you added system-wide packages, they will be inherited here."
    echo -e "There once you'll reuse the venv from a fresh environment it won't be able to find the packages installed system-wide.${NC}"

    read -p "I acknowledge and want to proceed. Type 'yes' to continue: " confirm_venv
    if [[ "$confirm_venv" != "yes" ]]; then
        echo "Aborting."
        exit 0
    fi

    python -m venv \
        --system-site-packages \
        --symlinks \
        $VENV_DIR
    source "$VENV_DIR/bin/activate"

    pip install nvidia-ml-py
    pip install -e .
    pip install -e third-party/verl
    pip uninstall -y pynvml

    # Use for validation against environment mismatch during job launch
    echo "$ENVIRONMENT_EDF_IMAGE" > "$VENV_DIR/.edf_image"
fi

# Retrieve the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(realpath $(dirname "$SCRIPT_DIR"))"
VENV_DIR="$(realpath $VENV_DIR)"

# If report dir is absolute path, use it directly; if relative, make it relative to base dir
if [[ "$REPORT_DIR" = /* ]]; then
    REPORT_DIR="$REPORT_DIR"
else
    REPORT_DIR="$BASE_DIR/${REPORT_DIR:-reports/verl}"
fi
mkdir -p "$REPORT_DIR"

REPORT_STDOUT_FILE="$REPORT_DIR/verl-$(date +%Y%m%d-%H%M%S)-%j.out"
REPORT_STDERR_FILE="$REPORT_DIR/verl-$(date +%Y%m%d-%H%M%S)-%j.err"

# Display summary of parameters
echo ""
echo "==================================="
echo "Configuration Summary:"
echo "  Config file:        $CONFIG_NAME"
echo "  Environment file:   $ENVIRONMENT"
echo "  Experiment name:    $EXPERIMENT_NAME"
echo "  Number of nodes:    $NUM_NODES"
echo "  Report directory:   $REPORT_DIR"
echo "  Stdout file:        $REPORT_STDOUT_FILE"
echo "  Stderr file:        $REPORT_STDERR_FILE"
echo "  Timeout:            $TIMEOUT"
echo "==================================="
echo ""

# --------------------------------
# Prepare and execute the sbatch command
# --------------------------------
cmd="sbatch \
--job-name=$EXPERIMENT_NAME \
--output=$REPORT_STDOUT_FILE \
--error=$REPORT_STDERR_FILE \
--nodes=$NUM_NODES \
--time=$TIMEOUT \
--ntasks-per-node=1 \
--cpus-per-task=280 \
--gpus-per-node=4 \
--partition=normal \
--mem=380G \
-A a127 \
--environment=$ENVIRONMENT \
--export=SCRIPT_DIR=$SCRIPT_DIR,REPORT_STDOUT_FILE=$REPORT_STDOUT_FILE,REPORT_STDERR_FILE=$REPORT_STDERR_FILE,VENV_DIR=$VENV_DIR \
\
${SCRIPT_DIR}/sbatch_ray_launcher.sh \
mm verl \
-c $CONFIG_NAME \
trainer.nnodes=$NUM_NODES \
trainer.n_gpus_per_node=4 \
trainer.experiment_name=$EXPERIMENT_NAME \
"
echo "$cmd"

# Confirm with the user before proceeding
if [ -z "$NO_CONFIRM" ]; then
    read -p "Proceed with these settings? (y/n): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo "Aborting."
        exit 0
    fi
fi

# Execute the command
$cmd




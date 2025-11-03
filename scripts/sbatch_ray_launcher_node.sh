#!/bin/bash

# Fail if any command fails
set -e

unset ROCR_VISIBLE_DEVICES # For some obscure reason this is set by CSCS environemtn
unset {HTTP,HTTPS,FTP,NO}_PROXY # Same here, having those variable set (even empty) can mess up with some tools (looking at you curl)
unset {http,https,ftp,no}_proxy # Same here, having those variable set (even empty) can mess up with some tools (looking at you curl)

echo "Activating virtual environment at $VENV_DIR"
source "$VENV_DIR/bin/activate"

# Verify environment image used
if [ ! -z "$SLURM_EDF_EXPANDED" ]; then
    CURRENT_EDF_IMAGE=$(echo "$SLURM_EDF_EXPANDED" | grep -oP '\s*image\s*=\s*"\K[^"]+')

    echo "Current EDF image: $CURRENT_EDF_IMAGE"

    # Load the edf image stored under $VENV_DIR/.edf_image
    if [ -f "$VENV_DIR/.edf_image" ]; then
        EXPECTED_EDF_IMAGE=$(cat "$VENV_DIR/.edf_image")
        echo "Expected EDF image: $EXPECTED_EDF_IMAGE"  
        if [ "$CURRENT_EDF_IMAGE" != "$EXPECTED_EDF_IMAGE" ]; then
            echo "Error: Current EDF image ($CURRENT_EDF_IMAGE) does not match expected EDF image ($EXPECTED_EDF_IMAGE)."
            exit 1
        else
            echo "EDF image verification passed."
        fi
    fi
else
    echo "SLURM_EDF_EXPANDED is not set, cannot verify EDF image."
    CURRENT_EDF_IMAGE=""
fi

# Traveller be aware, beneath lies the sources of all evil,
# What maketh the mighty nsjail to falter and succomb,
# Is a poison made of bashisms and slurmisms,
# A concoction so vile, that none who drinketh it,
# Shall evermore be able to run their scripts as intended.
# Beware traveller, and turn back now, lest ye be doomed forevermore.
rm /usr/lib64/libnl-3.so.200 # Remove libnl-3 that is symlinked to the one overwritten by SLURM
ln -s /usr/lib64/libnl-3.so.200.26.0 /usr/lib64/libnl-3.so.200 # Recreate symlink to the correct version
echo "Libnl-3 sha afer fix: $(sha256sum /usr/lib64/libnl-3.so.200  | awk '{print substr($1,1,16)}'), should be '2f233046cabcd5e7'"

# Summary
echo "Number of nodes: $SLURM_NNODES"
echo "Node list: $SLURM_NODELIST"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Memory per node: $SLURM_MEM_PER_NODE"
echo "Python Version: $(python --version)"
echo "Python path: $(python -c 'import sys; print(sys.executable)')"
echo "Ray path: $(which ray)"
echo "Working directory: $(pwd)"
echo "Environment variables: "

# Start ray HEAD node
if [[ $SLURM_NODEID -eq 0 ]]; then
    echo "Starting Ray HEAD node on $(hostname)"
    ray start \
        --head \
        --port=6379 \
        --log-color=false \
        --node-ip-address=$(hostname -i) \
        --num-cpus=$SLURM_CPUS_PER_TASK \
        --num-gpus=$SLURM_GPUS_ON_NODE \
        --include-dashboard=true \
        --dashboard-host="0.0.0.0" \
        --dashboard-port=8265

    # Await for the head node to initialize (await for 1 whole minues)
    trials=12
    for i in $(seq 1 $trials); do
        echo "Checking if Ray HEAD node is up (trial $i/$trials)..."
        num_nodes_up=$(ray list nodes --filter state=ALIVE --format json | python -c "import json; print(len(json.loads(input())))")
        if [[ $num_nodes_up -ge $SLURM_NNODES ]]; then
            echo "All $num_nodes_up nodes are up!"
            break
        else
            echo "Only $num_nodes_up/$SLURM_NNODES nodes are up. Retrying in 5 seconds..."
            sleep 5
        fi
    done

    # If after all trials the nodes are not up, exit with error
    if [[ $num_nodes_up -lt $SLURM_NNODES ]]; then
        echo "Error: Only $num_nodes_up/$SLURM_NNODES nodes are up after waiting. Exiting."
        ray stop
        exit 1
    fi
else
    # Sleep to ensure the head node starts before workers try to connect
    sleep 2

    # Start ray WORKER nodes
    echo "Starting Ray WORKER node on $(hostname)"
    ray start \
        --address=$(scontrol show hostnames $SLURM_NODELIST | head -n 1):6379 \
        --block \
        --log-color=false \
        --node-ip-address=$(hostname -i) \
        --num-cpus=$SLURM_CPUS_PER_TASK \
        --num-gpus=$SLURM_GPUS_ON_NODE
    exit 0 # Never run training script on worker nodes
fi

# Script only runs on the head node from here
ray status

SCRIPT_COMMAND=$(echo "$@" | sed "s|SLURM_JOB_ID|$SLURM_JOB_ID|g")

echo "Running training script on $(hostname)"
echo "Running command: $SCRIPT_COMMAND"
$SCRIPT_COMMAND # Execute the training script passed as an argument

# Finally stop the ray cluster (this should also stop all of the worker noeds)
ray stop

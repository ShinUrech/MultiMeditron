export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="online"

unset {HTTP,HTTPS,FTP,NO}_PROXY
unset {http,https,ftp,no}_proxy
unset ROCR_VISIBLE_DEVICES

if [[ $(id -u) == 30156 ]]; then
    export WANDB_MODE="offline"
    SESSION_NAME="cluster"

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        tmux attach -t "$SESSION_NAME"
        exit 0
    fi

    # Attempt removing bad eof
    rm /usr/lib64/libnl-3.so.200 # Remove libnl-3 that is symlinked to the one overwritten by SLURM
    ln -s /usr/lib64/libnl-3.so.200.26.0 /usr/lib64/libnl-3.so.200 # Recreate symlink to the correct version
    echo "Libnl-3 sha afer fix: $(sha256sum /usr/lib64/libnl-3.so.200  | awk '{print substr($1,1,16)}'), should be '2f233046cabcd5e7'"

    # Start a new tmux session (detached)
    tmux set-option -g default-shell /bin/bash
    tmux new-session -d -s "$SESSION_NAME"

    tmux split-window -v -t "$SESSION_NAME"
    tmux send-keys -t "$SESSION_NAME":0.0 'code tunnel --name=cluster-tunnel --cli-data-dir ~/.vscode-cli' C-m
    # tmux send-keys -t "$SESSION_NAME":0.1 'pip install -e . && pip install -e third-party/verl/' C-m
    tmux send-keys -t "$SESSION_NAME":0.1 'source ./.venv/bin/activate' C-m

    # Attach to the tmux session
    tmux select-pane -t "$SESSION_NAME":0.0
    tmux attach -t "$SESSION_NAME"
else
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TORCH_NCCL_AVOID_RECORD_STREAMS=1
    export NCCL_TIMEOUT=900  # seconds
    export NCCL_IB_HCA=mlx5_bond_
    export NCCL_SOCKET_NTHREADS=4
    export NCCL_NSOCKS_PERTHREAD=$RUNAI_NUM_OF_GPUS
fi

#!/bin/bash

RESET_COL="\033[0m"
RED_COL="\033[0;31m"
GREEN_COL="\033[0;32m"
YELLOW_COL="\033[0;33m"
BLUE_COL="\033[0;34m"

print_error() {
    echo -e "${RED_COL}[ERROR] $1${RESET_COL}"
}

print_success() {
    echo -e "${GREEN_COL}[SUCCESS] $1${RESET_COL}"
}

print_warning() {
    echo -e "${YELLOW_COL}[WARNING] $1${RESET_COL}"
}

print_info() {
    echo -e "${BLUE_COL}[INFO] $1${RESET_COL}"
}

confirm_operation() {
    read -p "Are you sure you want to proceed? (y/n): " choice
    case "$choice" in 
      y|Y ) return 0;;
      n|N ) print_info "Operation cancelled by user."; return 1;;
      * ) print_warning "Invalid input. Please enter 'y' or 'n'."; confirm_operation;;
    esac
}

print_warning "This script will attempt to fix cache issues (notably removing torch_extensions, triton, and other related caches)."
print_warning "This is potentially destructive and may lead to loss of data, ensure you are understanding the implications."
if ! confirm_operation; then
    exit 1
fi

# Removing entire cache directories (at ~/.cache)
TARGETS=(
    "~/.cache"
    "~/.ipython"
    "~/.local/share/virtualenv"
    "~/.local/lib/python*"
    "/tmp/*"
    "/capstor/store/cscs/swissai/a127/homes/$USER/hf*"
)

for TARGET in "${TARGETS[@]}"; do
    EXPANDED_TARGET=$(eval echo $TARGET)

    # If expanded target are multiple directories (like /tmp/*), handle them accordingly
    for DIR in $EXPANDED_TARGET; do
        if [ -e "$DIR" ]; then
            rm -rf $DIR
            if [ $? -eq 0 ]; then
                print_success "Successfully removed cache directory: $DIR"
            else
                print_error "Failed to remove cache directory: $DIR"
            fi
        else
            print_info "Cache directory does not exist, skipping: $DIR"
        fi
    done
done

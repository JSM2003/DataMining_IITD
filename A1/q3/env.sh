#!/bin/bash

set -e  # Exit immediately on error

# -------- Configuration --------
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
INSTALL_DIR="$HOME/miniconda3"
ENV_NAME="T11_DM_A1_q3"
INSTALLER="Miniconda3-latest-Linux-x86_64.sh"

# -------- Check for Miniconda --------
if [ -d "$INSTALL_DIR" ] && [ -f "$INSTALL_DIR/bin/conda" ]; then
    echo "Miniconda already installed at $INSTALL_DIR"
else
    echo "Miniconda not found. Installing Miniconda..."

    wget -q "$MINICONDA_URL" -O "$INSTALLER"
    bash "$INSTALLER" -b -p "$INSTALL_DIR"
    rm -f "$INSTALLER"

    echo "Miniconda installation completed."
fi

# -------- Initialize Conda --------
source "$INSTALL_DIR/etc/profile.d/conda.sh"

# Ensure conda is initialized for future shells
if ! grep -q "conda.sh" "$HOME/.bashrc"; then
    echo "Initializing conda in ~/.bashrc"
    conda init bash
fi

# -------- Check / Create Environment --------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
    conda activate "$ENV_NAME"
else
    echo "Creating Conda environment '$ENV_NAME'..."
    conda create -y -n "$ENV_NAME" python=3.10.19
    conda activate "$ENV_NAME"
    pip install numpy networkx tqdm
fi


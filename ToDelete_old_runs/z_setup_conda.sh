#!/bin/bash

# Step 1: Remove Miniconda Directory
echo "Removing existing Miniconda installation..."
rm -rf /p/home/jusers/cipolina-kun1/juwels/miniconda3/miniconda3
echo "Removal complete."

# Step 2: Download and Install Miniconda
echo "Downloading Miniconda installer..."
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

echo "Installing Miniconda on the linked dirrectory..."
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /p/scratch/laionize/cache-kun1/miniconda3

# Remove the installer
echo "Removing Miniconda installer..."
rm Miniconda3-latest-Linux-x86_64.sh

# Initialize Miniconda (optional, if needed for your system)
source /p/home/jusers/cipolina-kun1/juwels/miniconda3/bin/activate


# Step 3: Rebuild Environment from environment.yml
echo "Rebuilding Conda environment from environment.yml..."
conda env create -f /p/home/jusers/cipolina-kun1/juwels/coalitions/environment.yml


# Verify Conda Installation
echo "Verifying Conda installation..."
conda --version

echo "Conda environment setup complete."

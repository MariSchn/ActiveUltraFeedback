#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --partition=normal
#SBATCH --time=3:00:00
#SBATCH --container-writable
#SBATCH --job-name=build_container
#SBATCH --output=./logs/docker/build_%j.out

export PODMAN_STORAGE_CONF="/iopsstor/scratch/cscs/smarian/containers/podman_cache/config/storage.conf"

cd ~/ActiveUltraFeedback

echo "================= Build Script ================="
cat scripts/build_container.sh
echo -e "\n"

echo "================= Dockerfile ================="
cat Dockerfile
echo -e "\n"

echo "================= Building Docker image ================="
# Build the container using Podman
podman build -t activeuf_dev:latest .

# Remove old (test) container if it exists
rm -f $SCRATCH/containers/activeuf_dev.sqsh

# Convert the Podman image to a SquashFS image
enroot import -o $SCRATCH/containers/activeuf_dev.sqsh podman://activeuf_dev:latest

# Set permissions for the sqsh file
cd $SCRATCH/containers/
setfacl -b activeuf_dev.sqsh
chmod +r activeuf_dev.sqsh

# Test container
echo "================= Testing container with one GPU ================="
srun --environment=activeuf_dev python -m scripts.test_docker_one_gpu

echo "================= Testing container with multiple GPUs ================="
srun --environment=activeuf_dev python -m scripts.test_docker_multi_gpu
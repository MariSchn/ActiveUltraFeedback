#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --partition=normal
#SBATCH --time=3:00:00
#SBATCH --container-writable
#SBATCH --job-name=build_container
#SBATCH --output=./logs/docker/build_%j.out

cd ~/ActiveUltraFeedback

echo "================= Build Script ================="
cat scripts/docker/build_container.sh
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
srun --environment=activeuf_dev python -m scripts.docker.test_single_gpu

echo "================= Testing container with multiple GPUs ================="
srun --environment=activeuf_dev python -m scripts.docker.test_multi_gpu
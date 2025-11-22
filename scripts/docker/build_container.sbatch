#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --partition=normal
#SBATCH --time=3:00:00
#SBATCH --container-writable
#SBATCH --job-name=build_container
#SBATCH --output=./logs/docker/build_%j.out

CONTAINER_NAME=activeuf_dev

echo "================= Build Script ================="
cat scripts/docker/build_container.sh
echo -e "\n"

echo "================= Dockerfile ================="
cat Dockerfile
echo -e "\n"

echo "================= Building Docker image ================="
# Build the container using Podman
podman build -t $CONTAINER_NAME:latest .

# Remove old (test) container if it exists
rm -f $SCRATCH/containers/activeuf_dev.sqsh
enroot import -o $SCRATCH/containers/$CONTAINER_NAME.sqsh podman://$CONTAINER_NAME:latest

# Set permissions for the sqsh file
setfacl -b $SCRATCH/containers/$CONTAINER_NAME.sqsh
chmod +r $SCRATCH/containers/$CONTAINER_NAME.sqsh

# Test container
# echo "================= Testing container with one GPU ================="
srun --environment=activeuf_dev python -m scripts.test_docker_one_gpu

# echo "================= Testing container with multiple GPUs ================="
srun --environment=activeuf_dev python -m scripts.test_docker_multi_gpu
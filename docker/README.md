# Using the Tensorflow Object Detection API with Docker

This folder contains Docker images for both the CPU and GPU installations for the Tensorflow Object Detection API.

## GPU

To run Tensorflow Object Detection API with a GPU you need a NVIDIA graphics card. To be able to access a graphics card with Docker you also need to install [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker).

## CPU

If you don't have a GPU or you don't have a NVIDIA GPU you'll have to run the CPU version, which uses [Ubuntu](https://hub.docker.com/_/ubuntu) as the base image instead of [Nvidia-Cuda](https://hub.docker.com/r/nvidia/cuda). For this container you don't need [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker).

## Automatically choose between GPU/CPU

I also created a shell file that automatically detects if you have a NVIDIA GPU and automatically chooses between the CPU and GPU version.

You can run it by executing:
```bash
./start.sh
```

## Run an container

To run a container you need to move into the right  directory and then execute ```docker-compose up```.

After running the command docker should automatically download and install everything needed for the Tensorflow Object Detection API and open Jupyter on port 8888. If you also want to have access to the bash for training models you can simply say ```docker exec -it CONTAINER_ID```. For more information check out [Dockers documentation](https://docs.docker.com/).
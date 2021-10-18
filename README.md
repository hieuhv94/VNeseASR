## Install docker-nvidia toolkit
Install docker-nvidia at [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

## Build docker image
```
docker build -t <repo>/<image>:<tag> .
```

## Run container with GPU
```
docker run -it --runtime=nvidia --name w2l <repo>/<image>:<tag>
```

## Run container without GPU
```
docker run -it --name w2l <repo>/<image>:<tag>
```
## Install Docker CE and Nvidia Docker 2.0

#### Uninstall old version
``` sh
sudo apt-get remove docker docker-engine docker.io containerd runc
``` 
#### Install Docker CE
```sh

sudo apt-get update

sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io

#Verify that Docker CE is installed correctly by running the hello-world image
sudo docker run hello-world

```
#### Remove docker-nvidia 1.0
```sh
# If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo apt-get purge -y nvidia-docker
```
### Install docker-nvidia 2.0

#### Add the package repositories

```sh

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

```
#### Install nvidia-docker2 and reload the Docker daemon configuration
```sh
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
```

#### Test nvidia-smi with the latest official CUDA image
```sh
docker run --runtime=nvidia --rm nvidia/cuda:9.0-devel nvidia-smi
```

### Build docker image and run

#### To Build Image 
```sh
# build or pull image from dockerhub repo.
# If changes made to dockerfiles/Dockerfile.base file then re-build the base image otherwise build the library
docker build -f dockerfiles/Dockerfile.base -t mmdetection:base .
# Build  mmdetection library with latest tag
docker build -t mmdetection:latest .
```

#### To run 

```sh

# replace /home/ubuntu/code/ with path to the code directory

docker run -p 3000:3000 -v /home/ubuntu/code/:/home -it --runtime=nvidia --rm mmdetection:latest

cd mmdetection
python setup.py develop
python tools/test.py configs/retinanet_x101_64x4d_fpn_1x.py \
    checkpoints/retinanet_x101_64x4d_fpn_1x_20181218-a0a22662.pth \
    --out results.pkl --eval bbox

```

### Run Jupyter notebook

```ssh
ssh -L 3000:localhost:3000 ubuntu@[server-ip]
# run container bash and start jupyter notebook
nohup jupyter notebook --ip 0.0.0.0 --port 3000 --no-browser --allow-root &

```
<!-- scp -r /home/nasir/Downloads/download.tar ubuntu@172.31.20.106:code/mmdetection/data/ -->

<!-- ./tools/dist_train.sh configs/retinanet_x101_64x4d_fpn_1x.py 8 --validate  -->

For more details on Docker CE installation [docker-ce](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

For more details on Docker Nvidia installation [docker-nvidia](https://github.com/NVIDIA/nvidia-docker)
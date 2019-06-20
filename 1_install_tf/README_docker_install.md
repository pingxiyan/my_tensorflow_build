# Install tenforflow based on docker
Learn tensorflow, recommand refer: /tensorflow_learn/example/mnist/keras_method/mnist_lenet.py

#### install docker on Ubuntu.

	Uninstall older docker
	Refer: https://docs.docker.com/v17.12/install/linux/docker-ce/ubuntu/#set-up-the-repository

	$ sudo apt-get remove docker docker-engine docker.io containerd runc

	$ sudo apt-get update
	$ sudo apt-get install apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
    $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    $ sudo apt-key fingerprint 0EBFCD88
    $ sudo add-apt-repository \
		"deb [arch=amd64] https://download.docker.com/linux/ubuntu \
		$(lsb_release -cs) \
		stable"

	$ sudo apt-get update
	$ sudo apt-get install docker-ce

	$ docker --version
		Docker version 18.09.6, build 481bc77

	# Uninstall docker:
	$ sudo apt-get purge docker-ce
	$ sudo rm -rf /var/lib/docker

	If you can't pull docker project, you maybe need to setprox as follow.
**Note:Proxy set** 

	1.	Create a systemd drop-in directory for the docker service:
	$ sudo mkdir -p /etc/systemd/system/docker.service.d
	2.	Create a file called /etc/systemd/system/docker.service.d/http-proxy.conf that adds the HTTP_PROXY environment variable:
	[Service]
	Environment="HTTP_PROXY=http://proxy.example.com:80/"

	Or, if you are behind an HTTPS proxy server, create a file called /etc/systemd/system/docker.service.d/https-proxy.conf that adds the HTTPS_PROXY environment variable:
	[Service]
	Environment="HTTPS_PROXY=https://proxy.example.com:443/"
	3.	If you have internal Docker registries that you need to contact without proxying you can specify them via the NO_PROXY environment variable:
	[Service]    
	Environment="HTTP_PROXY=http://proxy.example.com:80/" "NO_PROXY=localhost,127.0.0.1,docker-registry.somecorporation.com"

	Or, if you are behind an HTTPS proxy server:
	[Service]    
	Environment="HTTPS_PROXY=https://proxy.example.com:443/" "NO_PROXY=localhost,127.0.0.1,docker-registry.somecorporation.com"
	4.	Flush changes:
	$ sudo systemctl daemon-reload
	5.	Restart Docker:
	$ sudo systemctl restart docker
	6.	Verify that the configuration has been loaded:
	$ systemctl show --property=Environment docker
	Environment=HTTP_PROXY=http://proxy.example.com:80/

	Or, if you are behind an HTTPS proxy server:
	$ systemctl show --property=Environment docker
	Environment=HTTPS_PROXY=https://proxy.example.com:443/

#### Install Tensorflow by docker

	1. Install nvidia docker.
		1.1 Refer: https://github.com/NVIDIA/nvidia-docker, Quickstart part.
			# If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
			docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
			sudo apt-get purge -y nvidia-docker

			# Add the package repositories
			curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
			  sudo apt-key add -
			distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
			curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
			  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
			sudo apt-get update

			# Install nvidia-docker2 and reload the Docker daemon configuration
			sudo apt-get install -y nvidia-docker2
			sudo pkill -SIGHUP dockerd

			# Test nvidia-smi with the latest official CUDA image
			docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi

	2. install tersorflow 
		$ docker pull tensorflow/tensorflow 			# latest stable version. default cpu version, don't need to install nvidia docker.
		$ docker pull tensorflow/tensorflow:1.13.1-gpu 	# specific version.
		$ docker pull tensorflow/tensorflow:1.13.1-gpu-py3

#### Start tensorflow in docker

	$ docker run -it tensorflow/tensorflow bash 

	map a local path to docker
	$ docker run --rm --network host -v [local path]:[docker path] -it tensorflow/tensorflow bash
	# Start GPU version
	$ docker run --rm --runtime=nvidia --network host -v [local path]:[docker path] -it tensorflow/tensorflow bash
	$ docker run --rm --network host --runtime=nvidia -it -v ~/mydisk2/mygithub/tensorflow_learn:/tensorflow_learn [container] bash

	Some parameters
	--rm: stop and exit, rm container
	--network host: using host's network, but proxy must be set in docker again.

	tensorflow's some dependencies
	$ pip install opencv-python
	$ apt-get install libglib2.0-0
	$ apt-get install libsm6
	$ apt-get install libxrender1
	$ apt-get install -y python-qt4
	$ apt-get install python-matplotlib

#### Common docker command 

	$ docker image ls --all 		# check all images
	$ docker image rm hello-world	# delete a docker image.
	$ docker container ls --all		# check all containers
	$ docker container stop $(container id)	# stop a container
	$ docker container rm $(container id) 	# delete container

	$ docker ps -l 		# check all docker process = docker container ls --all	
	$ docker commit [container id] [new container name] 	# Save your modification to new container. 

	$ docker exec -it [container id] bash		# Open multiple terminals in same docker.
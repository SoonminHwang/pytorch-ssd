USER_NAME=$(whoami)
nvidia-docker run -it -u ${USER_NAME} \
	-p $1:$1 \
	-p $2:$2 \
	-v /home/${USER_NAME}/workspace:/home/${USER_NAME}/workspace \
	-v /raid:/raid \
	-v /usr/share/zoneinfo:/usr/share/zoneinfo \
	-e NVIDIA_VISIBLE_DEVICES=5 \
	--shm-size=128G \
	--name $(whoami) \
	${USER_NAME}/pytorch-1.0:cuda10.0-cudnn7-dev-ubuntu16.04


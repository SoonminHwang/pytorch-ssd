USER_NAME=$(whoami)
nvidia-docker build -t \
	${USER_NAME}/pytorch-1.0:cuda10.0-cudnn7-dev-ubuntu16.04 \
	--build-arg UID=$(id -u) \
	--build-arg USER_NAME=${USER_NAME} \
	.

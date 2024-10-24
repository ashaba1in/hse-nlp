build:
	docker build \
	-t "hse-nlp-2024:latest" \
	-f Dockerfile \
	.

run:
	docker run \
	--gpus all \
	--network=host \
	--ipc=host \
	--mount type=bind,source=/home/$$(whoami)/hse-nlp/2024/,target=/srv/www/hse-nlp/ \
	hse-nlp-2024:latest

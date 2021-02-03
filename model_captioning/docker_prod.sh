docker build -t pytorch2009captioning_playground -f prod.Dockerfile .
docker run --rm --name captioning --ipc=host --gpus all -p 5006:5006 -dit -v `pwd`/project:/app pytorch2009captioning_playground

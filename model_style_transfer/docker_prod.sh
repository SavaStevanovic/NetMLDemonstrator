docker build -t pytorch2201style_playground -f prod.Dockerfile .
docker run --rm --name style --ipc=host --gpus all -p 5009:5009 -dit -v `pwd`/project:/app pytorch2201style_playground

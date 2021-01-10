docker build -t pytorch2009playground -f prod.Dockerfile .
docker run --rm --name keypoint --ipc=host --gpus all -p 5004:5004 -dit -v `pwd`/project:/app pytorch2009playground

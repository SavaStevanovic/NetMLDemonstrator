docker build -t pytorch2001playground - < prod.Dockerfile
docker run --rm --name detection --ipc=host --gpus all -p 5001:5001 -dit -v `pwd`/project:/app pytorch2001playground

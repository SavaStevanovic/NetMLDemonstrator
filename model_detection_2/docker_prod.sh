docker build -t pytorch2001detection_playground - < prod.Dockerfile
docker run --rm --name detection_2 --ipc=host --gpus all -p 5001:5001 -dit -v `pwd`/project:/app pytorch2001detection_playground

docker build -t pytorch2012segmentation_playground - < prod.Dockerfile
docker run --rm --name segmentation --ipc=host --gpus all -p 5005:5005 -dit -v `pwd`/project:/app pytorch2012segmentation_playground

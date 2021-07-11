docker build -t pytorch2009segmentation_playground -f prod.Dockerfile .
docker run --rm --name segmentation --ipc=host --gpus all -p 5005:5005 -dit -v `pwd`/project:/app pytorch2009segmentation_playground

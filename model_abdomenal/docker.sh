docker build -t pytorch2305abdomenal_playground .
xhost + 
docker run --rm --name pytorch2305abdomenal_playground -e DISPLAY=$DISPLAY --ipc=host --gpus all -p 6008:6006 -p 5005:5005 -dit -v `pwd`/project:/app -v /mnt/FastData/Data/segmentation/rsna-2023-abdominal-trauma-detection:/Data -v /tmp/.X11-unix:/tmp/.X11-unix  pytorch2305abdomenal_playground

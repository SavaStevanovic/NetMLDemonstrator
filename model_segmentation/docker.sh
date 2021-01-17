docker build -t pytorch2009segmentation_playground .
xhost + 
docker run --rm --name segmentation -e DISPLAY=$DISPLAY --ipc=host --gpus all -p 6008:6006 -p 5005:5005 -dit -v `pwd`/project:/app -v /home/sava/Documents/Data:/Data -v /tmp/.X11-unix:/tmp/.X11-unix  pytorch2009segmentation_playground

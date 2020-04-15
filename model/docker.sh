docker build -t pytorch2001playground .
xhost + 
docker run -e DISPLAY=$DISPLAY --ipc=host --gpus all -p 6006:6006 -it -v `pwd`/project:/app -v /media/sava/Data4/Coco:/Data/Coco -v /tmp/.X11-unix:/tmp/.X11-unix  pytorch2001playground

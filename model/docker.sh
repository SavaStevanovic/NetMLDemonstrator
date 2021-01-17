docker build -t pytorch2001playground .
xhost + 
docker run --rm --name detection -e DISPLAY=$DISPLAY --ipc=host --gpus all -p 6009:6009 -p 5001:5001 -dit -v `pwd`/project:/app -v /media/sava/Data/Coco:/Data/Coco -v /tmp/.X11-unix:/tmp/.X11-unix  pytorch2001playground

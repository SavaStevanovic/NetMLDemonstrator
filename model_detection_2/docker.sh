docker build -t pytorch2001detection_playground .
xhost + 
docker run --rm --name indoor -e DISPLAY=$DISPLAY --ipc=host --gpus all -p 6009:6009 -p 5010:5010 -dit -v `pwd`/project:/app -v /media/sava/Data/Data/detection/:/Data/detection -v /tmp/.X11-unix:/tmp/.X11-unix  pytorch2001detection_playground

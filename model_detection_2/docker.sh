docker build -t pytorch2001detection_playground .
xhost + 
docker run --rm --name detection_2 -e DISPLAY=$DISPLAY --ipc=host --gpus all -p 6009:6009 -p 5001:5001 -dit -v `pwd`/project:/app -v /media/sava/Data/detection:/Data/detection -v /tmp/.X11-unix:/tmp/.X11-unix  pytorch2001detection_playground

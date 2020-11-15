docker build -t pytorch2001playground .
xhost + 
docker run --rm --name detection -e DISPLAY=$DISPLAY --ipc=host --gpus all -p 6006:6006 -p 5001:5001 -dit -v `pwd`/project:/app -v /media/sava/Data1/Coco:/Data/Coco -v /tmp/.X11-unix:/tmp/.X11-unix  pytorch2001playground

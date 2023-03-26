docker build -t pytorch2201style_playground .
xhost + 
docker run --rm --name style -e DISPLAY=$DISPLAY --ipc=host --gpus all -p 6006:6006 -p 5009:5009 -dit -v `pwd`/project:/app -v /media/sava/Data/Data:/Data -v /tmp/.X11-unix:/tmp/.X11-unix  pytorch2201style_playground

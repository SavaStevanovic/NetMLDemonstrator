docker build -t pytorch2009captioning_playground .
xhost + 
docker run --rm --name captioning -e DISPLAY=$DISPLAY --ipc=host --gpus all -p 6009:6006 -p 5006:5006 -dit -v `pwd`/project:/app -v /home/sava/Documents/Data:/Data -v /tmp/.X11-unix:/tmp/.X11-unix  pytorch2009captioning_playground

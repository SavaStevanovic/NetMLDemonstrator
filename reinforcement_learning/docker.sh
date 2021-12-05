docker build -t pytorch2106rl_playground .
xhost + 
docker run --rm --name rl -e DISPLAY=$DISPLAY --ipc=host --gpus all -p 6011:6006 -p 5011:5011 -dit -v `pwd`/project:/app -v /tmp/.X11-unix:/tmp/.X11-unix  pytorch2106rl_playground

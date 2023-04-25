docker build -t pytorch2304rl_playground .
xhost + 
docker run --rm --name reinforcement_model --network="host" -e DISPLAY=$DISPLAY --ipc=host --gpus all -p 6011:6006 -p 5011:5011 -dit -v `pwd`/project:/app -v /tmp/.X11-unix:/tmp/.X11-unix  pytorch2304rl_playground

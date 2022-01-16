docker build -t pytorch2106rl_playground -f prod.Dockerfile .
xhost + 
docker run --rm --name reinforcement -e DISPLAY=$DISPLAY --ipc=host --gpus all -p 4322:4322 -p 5011:5011 -dit -v `pwd`/project:/app -v /tmp/.X11-unix:/tmp/.X11-unix pytorch2106rl_playground

docker build -t pytorch2001keypoint_playground .
xhost + 
docker run --rm --name keypoint -e DISPLAY=$DISPLAY --ipc=host --gpus all -p 6007:6007 -p 5004:5004 -dit -v `pwd`/project:/app -v /media/Data/keypoint:/Data/keypoint -v /tmp/.X11-unix:/tmp/.X11-unix  pytorch2001keypoint_playground

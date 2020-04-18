docker build -t cameraplayground .
xhost + 
docker run -e DISPLAY=$DISPLAY -p 5001:5001 -it -v `pwd`/project:/app -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0 cameraplayground

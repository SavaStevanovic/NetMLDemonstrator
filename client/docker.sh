docker build -t pytorch2001playground .
xhost + 
docker run -e DISPLAY=$DISPLAY -p 5000:5000 -it -v `pwd`/project:/app -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0 pytorch2001playground

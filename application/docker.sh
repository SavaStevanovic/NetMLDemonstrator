docker build -t node14playground .
xhost + 
docker run -e DISPLAY=$DISPLAY -p 4321:4321 -p 4200:4200 -it -v `pwd`/project:/app -v /tmp/.X11-unix:/tmp/.X11-unix  node14playground

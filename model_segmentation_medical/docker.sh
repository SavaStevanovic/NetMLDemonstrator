docker build -t pytorch2305segmentation_playground .
xhost + 
docker run --rm --name segmentation_medical -e DISPLAY=$DISPLAY --ipc=host --gpus all -p 6008:6006 -p 5005:5005 -dit -v `pwd`/project:/app -v /mnt/FastData/Data/segmentation/hubmap-hacking-the-human-vasculature:/Data -v /tmp/.X11-unix:/tmp/.X11-unix  pytorch2305segmentation_playground

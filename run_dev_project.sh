./docker_clean.sh
cd application 
./docker.sh
cd ..
cd model
./docker.sh
cd ..
cd model_keypoint
./docker.sh
cd ..
./docker_network.sh

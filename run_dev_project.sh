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
cd model_segmentation
./docker.sh
cd ..
cd reinforcement_learning
./docker.sh
cd ..
cd model_style_transfer
./docker.sh
cd ..
cd prometheus
./docker.sh
cd ..
cd grafana
./docker.sh
cd ..
cd nvidia
./docker.sh
cd ..
./docker_network.sh

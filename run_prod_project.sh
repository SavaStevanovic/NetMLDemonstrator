./docker_clean.sh
cd application 
./docker_prod.sh
cd ..
cd model
./docker_prod.sh
cd ..
cd model_keypoint
./docker_prod.sh
cd ..
cd model_segmentation
./docker_prod.sh
cd ..
cd reinforcement_learning
./docker_prod.sh
cd ..
cd prometheus
./docker_prod.sh
cd ..
cd grafana
./docker_prod.sh
cd ..
cd nvidia
./docker_prod.sh
cd ..
./docker_network.sh
sleep 3m
./deploy_site.sh

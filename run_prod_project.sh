./docker_clean.sh
cd application 
./docker_prod.sh
cd ..
./deploy_site.sh
cd model
./docker_prod.sh
cd ..
cd model_keypoint
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

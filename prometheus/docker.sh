docker volume create data
docker run --rm --name prometheus -d -p 9090:9090 -v data:/prometheus -v `pwd`/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus

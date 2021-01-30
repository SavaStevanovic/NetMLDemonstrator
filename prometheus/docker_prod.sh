docker volume create data_prod
docker run --rm --name prometheus -d -p 9090:9090 -v data_prod:/prometheus -v `pwd`/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus

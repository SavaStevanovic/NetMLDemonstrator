docker build -t nodeplayground - < prod.Dockerfile
docker run --rm --name menager -p 4321:4321 -p 4200:4200 -p 4320:4320 -dit -v `pwd`/project:/app nodeplayground

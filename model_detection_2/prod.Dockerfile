FROM nvcr.io/nvidia/pytorch:22.08-py3
RUN apt-get update && pip install --upgrade pip && apt-get install -y libsm6 libxext6 libxrender-dev
CMD pip install -r /app/requirements.txt; cd /app/; python3 service.py;

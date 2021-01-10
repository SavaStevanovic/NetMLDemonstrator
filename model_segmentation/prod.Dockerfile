FROM nvcr.io/nvidia/pytorch:20.09-py3
RUN apt-get update && pip install --upgrade pip && apt-get install -y libsm6 libxext6 libxrender-dev
EXPOSE 5005
COPY ./project/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
CMD python3 /app/service.py;

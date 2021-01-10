FROM nvcr.io/nvidia/pytorch:20.12-py3
RUN apt-get update && pip install --upgrade pip && apt-get install -y libsm6 libxext6 libxrender-dev
ENV QT_X11_NO_MITSHM=1
EXPOSE 5005
RUN apt-get -y install libprotobuf* protobuf-compiler ninja-build
COPY ./project/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
CMD pip uninstall -y tensorboard-plugin-dlprof; bash

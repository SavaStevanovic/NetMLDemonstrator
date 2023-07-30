FROM nvcr.io/nvidia/pytorch:23.03-py3
RUN apt-get update && apt-get install -y libtiff-dev
RUN pip install --upgrade pip 
ENV QT_X11_NO_MITSHM=1
RUN apt-get -y install libprotobuf* protobuf-compiler ninja-build
COPY ./project/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
CMD pip uninstall -y tensorboard-plugin-dlprof; bash

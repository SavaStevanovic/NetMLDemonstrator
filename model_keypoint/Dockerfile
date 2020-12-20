FROM nvcr.io/nvidia/pytorch:20.09-py3
RUN apt-get update && pip install --upgrade pip && apt-get install -y libsm6 libxext6 libxrender-dev
ENV QT_X11_NO_MITSHM=1
EXPOSE 5002
RUN apt-get -y install libprotobuf* protobuf-compiler ninja-build
CMD pip install -r requirements.txt; pip uninstall -y tensorboard-plugin-dlprof; bash

FROM nvcr.io/nvidia/pytorch:20.01-py3
RUN apt-get update && pip install --upgrade pip && apt-get install -y libsm6 libxext6 libxrender-dev
ENV QT_X11_NO_MITSHM=1
EXPOSE 5001
RUN apt-get -y install libprotobuf* protobuf-compiler ninja-build && git clone https://github.com/NVIDIA-AI-IOT/torch2trt && cd torch2trt && python setup.py install --plugins
CMD pip install -e /common/; pip install -r /app/requirements.txt; bash

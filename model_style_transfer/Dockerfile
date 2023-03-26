FROM nvcr.io/nvidia/pytorch:22.01-py3
RUN apt-get update && pip install --upgrade pip && apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx
ENV QT_X11_NO_MITSHM=1
RUN apt-get -y install libprotobuf* protobuf-compiler ninja-build
COPY ./project/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
RUN conda install -c conda-forge tensorboard
CMD pip uninstall -y tensorboard-plugin-dlprof; bash

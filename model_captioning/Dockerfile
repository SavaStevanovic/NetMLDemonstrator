FROM nvcr.io/nvidia/pytorch:23.01-py3
RUN apt-get update && pip install --upgrade pip && apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx
ENV QT_X11_NO_MITSHM=1
EXPOSE 5005
RUN apt -y install default-jre
RUN pip install git+https://github.com/Maluuba/nlg-eval.git@master
RUN nlg-eval --setup
RUN apt-get -y install libprotobuf* protobuf-compiler ninja-build 
COPY ./project/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
CMD pip uninstall -y tensorboard-plugin-dlprof; bash

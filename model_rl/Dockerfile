FROM nvcr.io/nvidia/pytorch:22.06-py3
RUN apt-get update 
ENV TZ=Europe/Belgrade
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install -y libglib2.0-0
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
COPY ./project/requirements.txt /app/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r /app/requirements.txt
RUN pip3 install gym[atari] 
RUN pip3 install gym[accept-rom-license]
RUN apt-get -y install -y python3-tk
RUN conda install pytorch
CMD bash

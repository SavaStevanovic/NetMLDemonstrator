FROM nvcr.io/nvidia/pytorch:22.06-py3
RUN apt-get update 
ENV TZ=Europe/Belgrade
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3-dev libglew-dev
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
COPY ./project/requirements.txt /app/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r /app/requirements.txt
RUN pip3 install gym[atari] 
RUN pip3 install gym[accept-rom-license]
RUN apt-get -y install -y python3-tk
RUN conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
RUN curl https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz --output mujoco210.tar.gz
RUN mkdir /root/.mujoco
RUN pip3 install -U gym[mujoco]
RUN pip3 install -U gym==0.23.1
RUN pip3 install -U 'mujoco-py<2.2,>=2.1'
RUN tar -xf mujoco210.tar.gz --directory /root/.mujoco
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
ENV D4RL_SUPPRESS_IMPORT_ERROR=1
CMD bash

FROM ubuntu:18.04
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6
RUN apt-get install -y python3 python3-pip python-opengl
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
COPY ./project/requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt
CMD bash

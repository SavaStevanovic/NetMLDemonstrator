FROM node:14.4.0
RUN apt-get -y update && apt-get -y install python-setuptools && easy_install pip && apt-get install -y libsm6 libxext6 libxrender-dev && npm install -g @angular/cli
RUN apt-get -y install python3-pip
RUN apt install -y libgl1-mesa-glx
RUN npm i sockjs-client
ENV QT_X11_NO_MITSHM=1
CMD pip3 install --upgrade pip; pip3 install -r /app/requirements.txt; cd /app/ml-site; bash

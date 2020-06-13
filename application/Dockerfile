FROM node:14.4.0
RUN apt-get -y update && apt-get -y install python-setuptools && easy_install pip && apt-get install -y libsm6 libxext6 libxrender-dev && npm install -g @angular/cli
ENV QT_X11_NO_MITSHM=1
CMD pip install -r /app/requirements.txt; bash

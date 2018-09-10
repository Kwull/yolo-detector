FROM kwull/yolo-docker
LABEL maintainer="Uladzimir Kazakevich"

WORKDIR /darknet
COPY /config/* /config/
COPY /darknet/* /darknet/

VOLUME /config
VOLUME /cctv/motion
VOLUME /cctv/tagged

CMD [ "python", "detect.py"]

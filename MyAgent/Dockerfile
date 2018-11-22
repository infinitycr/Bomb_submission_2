From python:3.6

ADD ./MyAgent /agent
ADD . /pommerman
RUN cd /pommerman && pip install .

EXPOSE 10080
ENV NAME Agent

WORKDIR /agent
ENTRYPOINT ["python"]
CMD ["run.py"]
FROM python:3.6

RUN apt-get -y update

ENV CODE_DIR /root/code

COPY . $CODE_DIR/sweet-rl
WORKDIR $CODE_DIR/sweet-rl

# Clean up pycache and pyc files
RUN rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip install -e .

CMD /bin/bash
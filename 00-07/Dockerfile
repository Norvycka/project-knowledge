# start from the image you already have
FROM python:3.12-slim

# all following steps happen in this folder inside the image
WORKDIR /app

# install numpy at build time, so it's part of the image
RUN pip install numpy

# copy your script from your PC into the image
COPY hello.py .

# what to run when a container starts
CMD ["python", "hello.py"]

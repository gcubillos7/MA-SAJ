FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git


RUN apt-get update
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && apt-get install -y python3.6 python3.6-dev
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN apt-get install -y python-apt --reinstall

#WORKDIR /

COPY ../requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN mkdir /install
WORKDIR /install

RUN pip3 install jsonpickle==0.9.6
# install Sacred (from OxWhirl fork)
RUN pip3 install setuptools
RUN git clone https://github.com/oxwhirl/sacred.git /install/sacred && cd /install/sacred && python3 setup.py install

RUN apt-get install -y htop iotop

# WORKDIR ./src
# MAMUJOCO
# RUN pip3 install git+https://github.com/schroederdewitt/multiagent_mujoco.git
# ## -- SMAC

ENV smac_ver 1
RUN pip3 install git+https://github.com/oxwhirl/smac.git
ENV SC2PATH /home/user/pymarl/3rdparty/StarCraftII

RUN apt-get install unzip
RUN mkdir -p /home/user/pymarl
COPY .. /home/user/pymarl

EXPOSE 8888
WORKDIR /home/user/pymarl


#WORKDIR /pymarl
#CMD line?
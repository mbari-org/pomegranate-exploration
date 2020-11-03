FROM centos:7

# need gcc
RUN yum -y groupinstall "Development Tools"

RUN yum -y install python3-devel \
 && pip3 install Cython pomegranate

RUN pip3 install ipython scikit-learn

RUN mkdir /opt/pomegranate \
          /opt/workspace

ENV PATH ${PATH:+$PATH:}/opt/pomegranate

WORKDIR  /opt/workspace

# No explicit entry point.

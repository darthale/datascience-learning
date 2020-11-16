ARG BASE_CONTAINER=jupyter/tensorflow-notebook

FROM $BASE_CONTAINER

LABEL author="Alessio Gastaldo"

USER root

# deep learning with keras

RUN pip install keras==2.4.3

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID

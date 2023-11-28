# This already contains the python, r, julia, latex, and nodejs runtimes
FROM jupyter/datascience-notebook@sha256:73a577b006b496e1a1c02f5be432f4aab969c456881c4789e0df77c89a0a60c2

RUN conda install --quiet --yes --freeze-installed -c conda-forge \
    'python-language-server' \
    'jupyterlab=3.6.0' \
    'r-languageserver' \
    'texlab' \
    'chktex' \
    'jupyter-lsp=2.2.0' \
  && jupyter labextension install --no-build \
    '@jupyter-lsp/jupyterlab-lsp@4.2.0' \
  && jupyter lab build --dev-build=False --minimize=True \
  && conda clean --all -f -y \
  && rm -rf \
    $CONDA_DIR/share/jupyter/lab/staging \
    /home/$NB_USER/.cache/yarn \
  && fix-permissions $CONDA_DIR \
  && fix-permissions /home/$NB_USER


#FROM jupyter/all-spark-notebook@sha256:a8de290d3f499123b23618a5811b2aacc969ad0675ac3b7562bed5d5e68c64d5

#RUN conda install --quiet --yes --freeze-installed -c conda-forge \
#    'python-language-server' \
#    'jupyterlab={JUPYTERLAB_VERSION}' \
#    'r-languageserver' \
#    'texlab' \
#    'chktex' \
#    'jupyter-lsp={JUPYTER_LSP_VERSION}' \
#  && jupyter labextension install --no-build \
#    '@jupyter-lsp/jupyterlab-lsp@{JUPYTERLAB_LSP_VERSION}' \
#  && jupyter lab build --dev-build=False --minimize=True \
#  && conda clean --all -f -y \
#  && rm -rf \
#    $CONDA_DIR/share/jupyter/lab/staging \
#    /home/$NB_USER/.cache/yarn \
#  && fix-permissions $CONDA_DIR \
#  && fix-permissions /home/$NB_USER

#ENV JUPYTER_RUNTIME_DIR=/tmp/.jupyter_runtime \
#    JUPYTERLAB_ENABLED=true

#ENV JUPYTERLAB_ENABLED=true

#RUN mamba install --quiet --yes \
#  'jupyterlab-lsp' \
#  'jedi-language-server' \
#  'nodejs>=18.0' \
#  &&  mamba clean --all -f -y
#
#RUN jlpm add --modules-folder /opt/conda/lib/node_modules/ typescript-language-server


# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
#ARG OWNER=jupyter
#ARG BASE_CONTAINER=$OWNER/pyspark-notebook
#FROM $BASE_CONTAINER
#
#LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"
#
## Fix: https://github.com/hadolint/hadolint/wiki/DL4006
## Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
#SHELL ["/bin/bash", "-o", "pipefail", "-c"]
#
#USER root
#
## RSpark config
#ENV R_LIBS_USER "${SPARK_HOME}/R/lib"
#RUN fix-permissions "${R_LIBS_USER}"
#
## R pre-requisites
#RUN apt-get update --yes && \
#    apt-get install --yes --no-install-recommends \
#    fonts-dejavu \
#    gfortran \
#    gcc && \
#    apt-get clean && rm -rf /var/lib/apt/lists/*
#
#USER ${NB_UID}
#
## R packages including IRKernel which gets installed globally.
#RUN mamba install --yes \
#    'r-base' \
#    'r-ggplot2' \
#    'r-irkernel' \
#    'r-rcurl' \
#    'r-sparklyr' && \
#    mamba clean --all -f -y && \
#    fix-permissions "${CONDA_DIR}" && \
#    fix-permissions "/home/${NB_USER}"

#
#RUN conda install --quiet --yes --freeze-installed -c conda-forge \
#    'python-language-server' \
#    'jupyterlab=3.6.0' \
#    'r-languageserver' \
#    'texlab' \
#    'chktex' \
#    'jupyter-lsp=2.2.0' \
#  && jupyter labextension install --no-build \
#    '@jupyter-lsp/jupyterlab-lsp@4.2.0' \
#  && jupyter lab build --dev-build=False --minimize=True \
#  && conda clean --all -f -y \
#  && rm -rf \
#    $CONDA_DIR/share/jupyter/lab/staging \
#    /home/$NB_USER/.cache/yarn \
#  && fix-permissions $CONDA_DIR \
#  && fix-permissions /home/$NB_USER
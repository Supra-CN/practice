# This already contains the python, r, julia, latex, and nodejs runtimes
FROM jupyter/all-spark-notebook

COPY ./.condarc $HOME/

RUN ls -alh $HOME/

RUN cat $HOME/.condarc

RUN conda info --all

# RUN conda install --quiet --yes --freeze-installed -c conda-forge \
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

# 切换到 root 用户以安装系统级依赖
USER root

# 备份并替换 apt 源列表为清华镜像源
RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list

# 更新 apt-get 并安装必要的工具
RUN apt-get update
RUN apt-get install -y nodejs npm curl

# 配置 pip 使用清华镜像源
RUN mkdir -p /root/.pip && \
    echo "[global]" > /root/.pip/pip.conf && \
    echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> /root/.pip/pip.conf

# 安装 Python 的 LSP 服务器和相关插件
RUN pip install jupyterlab jupyter-lsp \
    'python-lsp-server[all]' pylsp-mypy pyls-black pyls-isort jedi-language-server pyright
RUN pip install ChatTTS

RUN export NODE_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/nodejs-release/

# 配置 npm 使用清华镜像源
#RUN npm config set registry https://registry.npmmirror.com

# 安装其他语言的 LSP 服务器
RUN npm install -g bash-language-server
RUN npm install -g dockerfile-language-server-nodejs
RUN npm install -g javascript-typescript-langserver
#RUN npm install -g julia-language-server
RUN npm install -g sql-language-server
#RUN npm install -g texlab
RUN npm install -g typescript-language-server
RUN npm install -g unified-language-server
RUN npm install -g vscode-css-languageserver-bin
RUN npm install -g vscode-html-languageserver-bin
RUN npm install -g vscode-json-languageserver-bin
RUN npm install -g yaml-language-server

# 安装 R 语言服务器
RUN R -e "install.packages('languageserver', repos='http://cran.us.r-project.org')"

# 安装 Julia 语言服务器
RUN #julia -e 'import Pkg; Pkg.add("LanguageServer"); Pkg.add("SymbolServer")'

# 切换回默认用户
USER $NB_UID

# 安装 jupyterlab-lsp 扩展
#RUN jupyter labextension install @krassowski/jupyterlab-lsp
#
## 运行 JupyterLab
#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]


















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
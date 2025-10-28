# jupyter docker 使用方法

```shell
# 构建镜像 docker build -t ${image-name}，这里 image-name 是 supra-jupyter
docker build -t supra-jupyter .

# 运行容器 docker run -d --name ${container-name} -p ${host-port}:8888 -v "${PWD}":/home/jovyan/work ${image-name}
# -d 是后台运行
# --name ${container-name} 是设置容器名，这里用的是supra-jupyter-s1
# -p ${host-port}:8888 是宿主机端口，这里用的是18888
# -v "${PWD}":/home/jovyan/work，是目录映射，PWD 是本地目录，当前目录的绝对路径，也可以设置其他目录
# ${image-name} 是镜像名，这里是上面构建的supra-jupyter
docker run -d --name supra-jupyter-s1  -p 18888:8888 -v "${PWD}":/home/jovyan/work supra-jupyter

```
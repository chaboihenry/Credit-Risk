# BSase image: NVIDIA RAPIDS (Includes CUDA 12, cuDF, XGBoost-gpu)
# optimized for RTX 5080
FROM rapidsai/base:24.12-cuda12.5-py3.11

# set user to root to allow installing new packages
USER root

# set working directory
WORKDIR /app

# copy requirements file
COPY requirements.txt .

# install the auxiliary tools
RUN pip install --no-cache-dir -r requirements.txt

# expose the jupyter port
EXPOSE 8888

# Default Command: start jupyter lab automatically
CMD ["/bin/bash"]
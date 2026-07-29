# Banc d'essai d'inférence RF-DETR-Seg — image headless.
# Deux builds depuis le même fichier :
#   GPU : docker build -t archeologia-bench:gpu .
#   CPU : docker build -t archeologia-bench:cpu \
#           --build-arg BASE=python:3.11-slim-bookworm \
#           --build-arg ORT_PKG=onnxruntime==1.24.1 .
#
# onnxruntime-gpu est épinglé à la version exacte du venv CPU du plugin (1.24.1) pour que
# les noyaux et les optimisations de graphe correspondent. ORT >= 1.19 exige CUDA 12 + cuDNN 9,
# fournis par le tag 12.6.3-cudnn-runtime.
ARG BASE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
FROM ${BASE}

ARG ORT_PKG=onnxruntime-gpu==1.24.1

# python3.11 n'est pas dans les dépôts d'ubuntu 22.04 : deadsnakes. Sur la base python:*-slim
# le paquet est déjà là et apt-get échouerait — d'où le test d'existence.
RUN if [ ! -x /usr/local/bin/python3 ]; then \
      apt-get update && \
      apt-get install -y --no-install-recommends software-properties-common gnupg ca-certificates && \
      add-apt-repository -y ppa:deadsnakes/ppa && \
      apt-get update && \
      apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-distutils curl libglib2.0-0 libexpat1 && \
      curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
      ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
      ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
      apt-get purge -y software-properties-common gnupg curl && apt-get autoremove -y && \
      rm -rf /var/lib/apt/lists/*; \
    fi

# opencv et onnxruntime sont PINNÉS sur les versions du venv d'exécution du plugin
# (dev/runner_onnx/.venv_onnx). Le banc mesure la chaîne de production : si findContours
# ou resize changent de version majeure, ce n'est plus la même chaîne et la parité
# validée sur Windows ne vaut plus rien ici.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
      "numpy<3" \
      ${ORT_PKG} \
      opencv-python-headless==4.10.0.84 \
      pillow \
      pyyaml \
      pandas \
      "shapely>=2" \
      pyogrio \
      geopandas \
      pycocotools \
      scikit-image

# Le plugin est monté en lecture seule : interdire l'écriture de __pycache__.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONPATH=/plugin/src:/harness

WORKDIR /harness
CMD ["python3", "-m", "tools.bench", "--help"]

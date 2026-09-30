# Project container for the sign-language pipeline on the Jetson Orin Nano (L4T R36.4.x).
#
# Base = NGC PyTorch 25.06, **iGPU variant** (the tag the course image lpcv:fall2026 was built from;
# its layers are already on the board). Python 3.12.3, torch 2.8.0a0 (nv25.06), TensorRT 10.11.0.33,
# CUDA 12.9, trtexec. We add only what the repo scripts import.
#
# Do NOT use the plain `pytorch:25.06-py3` tag that is also pulled on the board: that is the SBSA
# (server ARM) build; its TensorRT refuses the Orin GPU ("Target GPU SM 87 is not supported").
#
# Build via jetson/run.sh build (passes UID/GID so bind-mounted files stay owned by you, not root).
FROM nvcr.io/nvidia/pytorch:25.06-py3-igpu

ARG UID=1000
ARG GID=1000
ARG UNAME=slt

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

# numpy stays <2 for the NVIDIA torch/tensorrt python bindings (SESSION-LOG §0).
# onnxruntime = CPU wheel, only for off-GPU verification of ONNX graphs.
# portalocker + rouge are imported by the vendored unisign/external_metrics/sacrebleu.py and
# unisign/metrics.py; without them eval_openasl.py dies at import, before any inference.
RUN python3 -m pip install --no-cache-dir \
        "numpy<2" \
        "opencv-python-headless>=4.8" \
        "onnx>=1.15" \
        "onnxruntime>=1.17" \
        "transformers>=4.45,<5" \
        sentencepiece \
        protobuf \
        accelerate \
        portalocker \
        rouge \
    && python3 -c "import numpy, cv2, onnx, onnxruntime, transformers, sentencepiece; print('numpy', numpy.__version__, 'cv2', cv2.__version__, 'transformers', transformers.__version__)" \
    && python3 -c "import portalocker, rouge; print('portalocker', portalocker.__version__)"

# Non-root user matching the host account. Explicit HOME/USER so getpass.getuser() and the
# torch inductor cache never crash on a UID with no passwd entry (Jetson-access.md).
RUN (getent group ${GID} >/dev/null || groupadd -g ${GID} ${UNAME}) \
    && useradd -m -u ${UID} -g ${GID} -G video,render -s /bin/bash ${UNAME} 2>/dev/null \
       || useradd -m -u ${UID} -g ${GID} -s /bin/bash ${UNAME}
ENV HOME=/home/${UNAME} \
    USER=${UNAME} \
    PYTHONPATH=/workspace \
    PYTHONUNBUFFERED=1

# Silence the NGC banner / licence text / driver-version lectures that the entrypoint prints on every
# `docker run` (they drown script output). The GPU-presence and shm checks are kept.
RUN rm -f /opt/nvidia/entrypoint.d/10-banner.sh /opt/nvidia/entrypoint.d/12-banner.sh \
          /opt/nvidia/entrypoint.d/15-container-copyright.txt /opt/nvidia/entrypoint.d/21-pytorch-copyright.txt \
          /opt/nvidia/entrypoint.d/30-container-license.txt /opt/nvidia/entrypoint.d/51-gpu-sm-version-check.sh \
          /opt/nvidia/entrypoint.d/52-gpu-driver-version-check.sh /opt/nvidia/entrypoint.d/54-cpu-capabilities-check.sh \
          /opt/nvidia/entrypoint.d/56-network-driver-version-check.sh

USER ${UNAME}
WORKDIR /workspace
CMD ["bash"]

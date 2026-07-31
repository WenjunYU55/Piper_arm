ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/tesseract
RUN /opt/tesseract/bin/pip install --no-cache-dir \
      numpy==2.2.6 scipy==1.15.2 PyYAML==6.0.2 \
      tesseract-robotics-nanobind==0.35.0.6

COPY piper_ros_foxy/src/piper_tesseract_foxy /src/piper_tesseract_foxy
RUN /opt/tesseract/bin/pip install --no-cache-dir --no-deps /src/piper_tesseract_foxy

COPY piper_ros_foxy/src/piper_description /models/share/piper_description
COPY L515_camera/calibration/hand_eye/session_20260701_local/calibration_result.yaml \
     /models/calibration_result.yaml
COPY piper_ros_foxy/src/piper_tesseract_foxy/model/piper.srdf /models/piper.srdf
COPY piper_ros_foxy/src/piper_tesseract_foxy/model/collision_model.yaml \
     /models/collision_model.yaml

RUN /opt/tesseract/bin/python -m piper_tesseract_foxy.model_builder \
      --xacro /models/share/piper_description/urdf/piper_description.xacro \
      --calibration /models/calibration_result.yaml \
      --manifest /models/collision_model.yaml \
      --output /models/share/piper_description/urdf/piper_planning.urdf

ENV TESSERACT_RESOURCE_PATH=/models/share
ENV PIPER_TESSERACT_URDF=/models/share/piper_description/urdf/piper_planning.urdf
ENV PIPER_TESSERACT_SRDF=/models/piper.srdf
ENV PIPER_TESSERACT_COLLISION_MANIFEST=/models/collision_model.yaml
ENV PIPER_TESSERACT_SPOOL=/spool

RUN useradd --uid 1000 --create-home planner
USER planner
ENTRYPOINT ["/opt/tesseract/bin/python", "-m", "piper_tesseract_foxy.worker"]

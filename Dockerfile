FROM nvidia/cuda:11.4.0-base-ubuntu20.04
FROM python:3.10

RUN groupadd -r bia_riken_jp && useradd -m --no-log-init -r -g bia_riken_jp bia_riken_jp

RUN mkdir -p /home/febrian/BIA_ensemble_ATLAS_2022 /input /output \
    && chown bia_riken_jp:bia_riken_jp /home/febrian/BIA_ensemble_ATLAS_2022 /input /output

USER bia_riken_jp

# set a directory for the app
WORKDIR /home/febrian/BIA_ensemble_ATLAS_2022

ENV PATH="/home/bia_riken_jp/.local/bin:${PATH}"

# RUN echo ${PATH}
# RUN echo ${PWD}

RUN python -m pip install --user -U pip

COPY --chown=bia_riken_jp:bia_riken_jp requirements.txt /home/febrian/BIA_ensemble_ATLAS_2022/
RUN python -m pip install --user -r requirements.txt

COPY --chown=bia_riken_jp:bia_riken_jp process.py /home/febrian/BIA_ensemble_ATLAS_2022/
COPY --chown=bia_riken_jp:bia_riken_jp settings.py /home/febrian/BIA_ensemble_ATLAS_2022/
COPY --chown=bia_riken_jp:bia_riken_jp grandchallenges/ /home/febrian/BIA_ensemble_ATLAS_2022/grandchallenges
COPY --chown=bia_riken_jp:bia_riken_jp trained_models/ /home/febrian/BIA_ensemble_ATLAS_2022/trained_models
COPY --chown=bia_riken_jp:bia_riken_jp test-images/ /home/febrian/BIA_ensemble_ATLAS_2022/test-images
# COPY --chown=bia_riken_jp:bia_riken_jp isles/ /home/febrian/BIA_ensemble_ATLAS_2022/isles

ENTRYPOINT python -m process $0 $@

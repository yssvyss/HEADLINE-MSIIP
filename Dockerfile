# python:3.10-slim + pip torch wheels is smaller than pytorch/pytorch (conda-based).
FROM --platform=linux/amd64 python:3.10-slim AS example-algorithm-amd64

# All paths here are relative to the build context - this can be set in do_build.sh.

ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN pip install --no-cache-dir uv

WORKDIR /opt/app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt /opt/app/requirements.txt
RUN uv pip install \
    --system \
    --no-cache \
    --requirement /opt/app/requirements.txt

# ── Participants' model code ─────────────────────────────────────────────────────────
COPY my-model-code/ /opt/app/my-model-code/
# Depending on your project setup, you could potentially install the model code as a 
#       package if it has a setup.py:
#   RUN uv pip install --system --no-cache --no-deps -e /opt/app/my-model-code/
# Or alternatively, add the model code to PYTHONPATH:
#   ENV PYTHONPATH=/opt/app/my-model-code:${PYTHONPATH}
# Or simply copy the model code as above, whatever works for your project.

# ── HuggingFace / torch cache if neeeded ─────────────────────────────────────────────
ENV HF_HOME=/tmp/hf_home
ENV HF_HUB_OFFLINE=1
ENV TORCH_HOME=/tmp/torch_home
ENV HEADLINE_SAVE_PROCESS_FILES=0

USER user

COPY --chown=user:user inference.py /opt/app/inference.py

ENTRYPOINT ["python", "inference.py"]

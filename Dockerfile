FROM pytorch/pytorch:2.11.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OBSCAL_ROOT=/opt/obscal

WORKDIR ${OBSCAL_ROOT}

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY assets ./assets

RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=${OBSCAL_ROOT}/src
ENTRYPOINT ["python"]
CMD ["-m", "obscal.cli.verify"]

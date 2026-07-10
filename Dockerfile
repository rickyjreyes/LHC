FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --requirement requirements.txt

COPY . .

WORKDIR /app/lhcb_wct_analysis_kit

RUN mkdir -p data data_control outputs

VOLUME ["/app/lhcb_wct_analysis_kit/data", "/app/lhcb_wct_analysis_kit/data_control", "/app/lhcb_wct_analysis_kit/outputs"]

CMD ["python", "run_all.py", "--dry-run"]

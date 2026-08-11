# Behavior AI serving image.
#
# Deliberately built from requirements-serve.txt, not requirements.txt: the
# inference path never imports torch / xgboost / matplotlib / anthropic, and
# torch alone would dominate the image size.
FROM python:3.12-slim

# libgomp1 is LightGBM's OpenMP runtime — the wheel links against it and import
# fails without it on slim images.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

# Application code only. Secrets arrive as environment variables at run time and
# are never baked into the image.
COPY app/ ./app/
COPY learning/ ./learning/
# 두 벌 다 넣는다. ConfigMap 의 PRODUCTION_MODEL_DIR 이 어느 쪽을 가리키든 이미지가
# 그것을 갖고 있어야 하고, 되돌릴 때도 이미지를 다시 굽지 않고 설정 한 줄로 끝난다.
COPY models/candidate/revalidation_two_view_participant_safe_20260722/ \
     ./models/candidate/revalidation_two_view_participant_safe_20260722/
COPY models/candidate/surface_aware_veto_20260810/ \
     ./models/candidate/surface_aware_veto_20260810/

RUN useradd --create-home --uid 10001 aiservice && chown -R aiservice:aiservice /app
USER aiservice

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Binds 0.0.0.0 inside the container; compose publishes it on 127.0.0.1 only,
# so nothing outside the host can reach it.
EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8010/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]

FROM python:3.12-slim
WORKDIR /app
COPY requirements-preprocessing.txt requirements-catboost.txt requirements-service.txt ./
RUN pip install --no-cache-dir -r requirements-service.txt
COPY mos_trans ./mos_trans
COPY artifacts/route_delay_trend_submission/catboost_residual_final.cbm ./artifacts/route_delay_trend_submission/catboost_residual_final.cbm
COPY artifacts/calibrated_probability/model.joblib ./artifacts/calibrated_probability/model.joblib
EXPOSE 8001
CMD ["uvicorn", "mos_trans.ml_api:app", "--host", "0.0.0.0", "--port", "8001"]

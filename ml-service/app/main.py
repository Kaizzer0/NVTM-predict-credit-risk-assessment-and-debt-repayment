from secrets import compare_digest

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from app import training_jobs
from app.config import get_settings
from app.schemas import PredictRequest, PredictResponse
from app.scoring import ModelUnavailable, metadata, predict, input_contract, sample_request, test_curves

app = FastAPI(
    title="Credit Risk ML Service",
    description=(
        "Ước lượng rủi ro gặp khó khăn thanh toán theo TARGET Home Credit Default Risk. "
        "Không dùng cho phê duyệt tín dụng thật."
    ),
    version="2.0.0-home-credit",
)


@app.exception_handler(ModelUnavailable)
def model_unavailable(_request: Request, exc: ModelUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
def health():
    try:
        ready = bool(metadata())
    except ModelUnavailable:
        ready = False
    return {"status": "ok", "service": "ml-service", "model_ready": ready}


@app.get("/model/info")
def model_info():
    return metadata()


@app.get("/model/curves")
def model_curves():
    return test_curves()


@app.get("/training")
def training_status():
    return training_jobs.status()


@app.post("/training", status_code=202)
def start_training(x_ml_training_token: str = Header(default="")):
    token = get_settings().ml_training_token
    if not token:
        raise HTTPException(status_code=503, detail="Huấn luyện qua API chưa được cấu hình")
    if not compare_digest(token.encode(), x_ml_training_token.encode()):
        raise HTTPException(status_code=403, detail="Không có quyền gọi dịch vụ huấn luyện")
    job = training_jobs.start()
    if job is None:
        raise HTTPException(status_code=409, detail="Đang có một lượt huấn luyện")
    return job


@app.get("/model/input-schema")
def model_input_schema():
    return input_contract()


@app.get("/model/sample")
def model_sample():
    return sample_request()


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest):
    return predict(payload)


@app.post("/validate")
def validate_inputs(payload: PredictRequest):
    from app.home_credit_features import CONTRACT_VERSION
    return {"inputs": payload.model_dump(), "contract_version": CONTRACT_VERSION}

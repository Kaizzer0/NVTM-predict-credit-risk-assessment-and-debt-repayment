from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.roles import ADMIN, OFFICER
from app.database import get_db
from app.dependencies import require_roles
from app.models.model_registry import ModelRegistry
from app.models.user import User
from app.schemas.model_registry import ModelRegistryPublic
from app.services.ml_client import request_home_credit

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/current")
def current_model(
    _: User = Depends(require_roles(ADMIN, OFFICER)),
):
    return request_home_credit("GET", "/model/info")


@router.get("/current/curves")
def current_curves(_: User = Depends(require_roles(ADMIN, OFFICER))):
    return request_home_credit("GET", "/model/curves")


@router.get("/training")
def training_status(_: User = Depends(require_roles(ADMIN, OFFICER))):
    return request_home_credit("GET", "/training")


@router.post("/training", status_code=202)
def start_training(_: User = Depends(require_roles(ADMIN))):
    return request_home_credit(
        "POST", "/training", headers={"X-ML-Training-Token": get_settings().ml_training_token},
    )


@router.get("", response_model=list[ModelRegistryPublic])
def list_models(
    db: Session = Depends(get_db),
    _: User = Depends(require_roles(ADMIN, OFFICER)),
):
    return db.query(ModelRegistry).order_by(ModelRegistry.created_at.desc()).all()

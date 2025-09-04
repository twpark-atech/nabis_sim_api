# app/api/v1/scenarios_bridge.py

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import Response
from fastapi.encoders import jsonable_encoder

from app.schemas import ScenarioCreate
from app.tasks.scenario_tasks import generate_and_persist

router = APIRouter()

@router.post("/", status_code=status.HTTP_202_ACCEPTED, response_class=Response)
def create_scenario(payload: ScenarioCreate) -> Response:
    generate_and_persist.delay(jsonable_encoder(payload))
    return Response(status_code=status.HTTP_202_ACCEPTED)
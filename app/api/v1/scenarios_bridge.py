# app/api/v1/scenarios_bridge.py
from __future__ import annotations

from fastapi import APIRouter, status, Depends
from fastapi.responses import Response
from fastapi.encoders import jsonable_encoder
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.schemas import ScenarioCreate
from app.db import get_db
from app.models import Scenario
from app.tasks.scenario_tasks import generate_and_persist

router = APIRouter()

@router.post("/", status_code=status.HTTP_202_ACCEPTED, response_class=Response)
def create_scenario(payload: ScenarioCreate, db: Session = Depends(get_db)) -> Response:
    # 트랜잭션 내에서 advisory lock을 잡고 MAX+1 채번 → 경쟁조건 방지
    with db.begin():
        db.execute(text("SELECT pg_advisory_xact_lock( hashtext('SIM_SCENARIOS')::bigint )"))
        next_id = db.execute(
            text('SELECT COALESCE(MAX("scenario_id"), 0) + 1 FROM "SIM_SCENARIOS"')
        ).scalar_one()

        sc = Scenario(
            scenario_id=next_id,
            name=payload.name,
            route_id=payload.route_id,
            headway_min=payload.headway_min,
            start_time=payload.start_time,
            end_time=payload.end_time,
            departure_time=payload.departure_time,
            path_type=payload.path_type,
            status="생성 중",
            # 결과 컬럼들은 Celery가 채움 (NULL 허용이어야 함)
        )
        db.add(sc)
    # 커밋 후 시나리오 ID로 비동기 처리 시작
    generate_and_persist.delay(jsonable_encoder(payload), next_id)
    return Response(status_code=status.HTTP_202_ACCEPTED)

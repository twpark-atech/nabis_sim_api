# app/api/v1/scenarios_bridge.py
from __future__ import annotations

from fastapi import APIRouter, status, Depends, HTTPException
from fastapi.responses import Response
from fastapi.encoders import jsonable_encoder
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.schemas import ScenarioCreate
from app.db import get_db
from app.models import Scenario
from app.tasks.scenario_tasks import generate_and_persist

router = APIRouter()

SCENARIO_ID_MAX = 100_000_000

@router.post("/", status_code=status.HTTP_202_ACCEPTED, response_class=Response)
def create_scenario(payload: ScenarioCreate, db: Session = Depends(get_db)) -> Response:
    # 트랜잭션 내 advisory lock으로 동시성 제어
    with db.begin():
        # 같은 리소스 이름으로 트랜잭션 락
        db.execute(text("SELECT pg_advisory_xact_lock( hashtext('SIM_SCENARIOS')::bigint )"))

        # 가장 작은 미사용 ID 찾기 (1부터 시작, 100,000,000 미만)
        next_id_sql = text(f"""
        WITH candidates AS (
            -- 1이 비어있으면 1
            SELECT 1 AS candidate
            WHERE NOT EXISTS (SELECT 1 FROM "SIM_SCENARIOS" WHERE "scenario_id" = 1)
            UNION ALL
            -- s+1이 비어있는 것들
            SELECT s."scenario_id" + 1 AS candidate
            FROM "SIM_SCENARIOS" s
            WHERE s."scenario_id" + 1 < :cap
              AND NOT EXISTS (
                  SELECT 1 FROM "SIM_SCENARIOS" z
                  WHERE z."scenario_id" = s."scenario_id" + 1
              )
        )
        SELECT COALESCE(MIN(candidate), 1) AS next_id
        FROM candidates
        """)
        next_id = int(db.execute(next_id_sql, {"cap": SCENARIO_ID_MAX}).scalar_one())

        if not (1 <= next_id < SCENARIO_ID_MAX):
            raise HTTPException(status_code=409, detail="No available scenario_id < 100,000,000")

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
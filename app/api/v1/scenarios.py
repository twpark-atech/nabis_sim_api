# app/api/v1/scenarios.py
from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Scenario
from app.schemas import ScenarioOut

router = APIRouter()

@router.get("/", response_model=List[ScenarioOut])
def list_scenarios(db: Session = Depends(get_db)):
    # [시나리오-1] DB에서 기존 시나리오 불러오기
    rows = db.execute(select(Scenario)).scalars().all()
    return rows

# [시나리오-2/3/4]는 다음 단계에서:
# - 노선 API 결과 기반 정류장 딕셔너리 생성
# - 정류장 좌표 → 경로 API 실행
# - 입력 파라미터(배차/운행시간/출발시간) → 100ms 단위 속도·위치 시뮬 결과
#   => /scenarios/simulate 같은 엔드포인트로 이어서 구현 예정

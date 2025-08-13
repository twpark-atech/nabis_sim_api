# app/api/v1/routes.py
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select, and_, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Route, Station
from app.schemas import RouteCreate, RouteOut, RouteStationsOut, StationInfo

router = APIRouter()

ROUTE_ID_MIN = 500_000_000   # 포함
ROUTE_ID_MAX = 600_000_000   # 미포함

def _next_virtual_route_id(db: Session) -> int:
    max_id = db.execute(
        select(func.max(Route.route_id)).where(
            and_(Route.route_id >= ROUTE_ID_MIN, Route.route_id < ROUTE_ID_MAX)
        )
    ).scalar()
    if max_id is None:
        return ROUTE_ID_MIN
    nxt = max_id + 1
    if nxt >= ROUTE_ID_MAX:
        raise HTTPException(status_code=409, detail="virtual route_id capacity exhausted")
    return nxt

def _canonical(lst: List[int]) -> tuple[int, ...]:
    # 순서는 무시하되, "개수와 종류"를 그대로 반영하기 위해 다중집합 비교용으로 정렬만 적용
    return tuple(sorted(lst))

def _find_duplicate_route(db: Session, station_list: List[int]) -> Route | None:
    # 같은 길이의 노선들만 후보로 가져온 뒤, 파이썬에서 다중집합 동일성(정렬) 비교
    n = len(station_list)
    candidates = db.execute(
        select(Route).where(func.cardinality(Route.station_list) == n)
    ).scalars().all()
    target = _canonical(station_list)
    for r in candidates:
        if _canonical(r.station_list or []) == target:
            return r
    return None

@router.post("/", response_model=RouteOut, status_code=status.HTTP_201_CREATED)
def create_virtual_route(payload: RouteCreate, response: Response, db: Session = Depends(get_db)):
    if not payload.station_list:
        raise HTTPException(status_code=400, detail="station_list must not be empty")

    # station_id 존재 여부 확인 (정확한 ID 기반)
    existing = set(
        db.execute(
            select(Station.station_id).where(Station.station_id.in_(payload.station_list))
        ).scalars().all()
    )
    missing = [sid for sid in payload.station_list if sid not in existing]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={"message": "one or more station_ids do not exist", "missing": missing},
        )

    # 1) 정류장ID 리스트가 (순서 무시, 개수/종류 동일)한 노선이 이미 있으면 → 생성하지 않고 기존 노선 반환(200)
    dup = _find_duplicate_route(db, payload.station_list)
    if dup:
        response.status_code = status.HTTP_200_OK
        return RouteOut(route_id=dup.route_id, route_name=dup.route_name, station_list=dup.station_list)

    # 2) 없으면 5억번대 route_id 순차 발급 후 생성 (경합 대비 재시도)
    for _ in range(5):
        rid = _next_virtual_route_id(db)
        try:
            row = Route(route_id=rid, route_name=payload.route_name, station_list=payload.station_list)
            db.add(row)
            db.commit()
            db.refresh(row)
            return RouteOut(route_id=row.route_id, route_name=row.route_name, station_list=row.station_list)
        except IntegrityError:
            db.rollback()
            continue

    raise HTTPException(status_code=409, detail="could not allocate a unique virtual route_id")


# ===== 조회용 (이미 있으면 유지) =====

@router.get("/{route_id}/stations", response_model=RouteStationsOut)
def get_route_stations(route_id: int, db: Session = Depends(get_db)):
    route = db.get(Route, route_id)
    if not route:
        raise HTTPException(status_code=404, detail="route not found")

    station_ids: List[int] = route.station_list or []
    if not station_ids:
        return RouteStationsOut(route_id=route.route_id, route_name=route.route_name, stations=[])

    rows = db.execute(select(Station).where(Station.station_id.in_(station_ids))).scalars().all()
    by_id = {s.station_id: s for s in rows}

    ordered = [
        StationInfo(station_id=sid, station_name=by_id[sid].station_name, x=by_id[sid].x, y=by_id[sid].y)
        for sid in station_ids
        if sid in by_id
    ]
    return RouteStationsOut(route_id=route.route_id, route_name=route.route_name, stations=ordered)

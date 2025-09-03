# app/api/v1/routes.py
from math import sqrt
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select, and_, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Route, Station
from app.schemas import RouteCreate, RouteOut

router = APIRouter()

ROUTE_ID_MIN = 500_000_000
ROUTE_ID_MAX = 600_000_000


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
        raise HTTPException(status_code=409, detail="route_id가 초과되었습니다.")
    return nxt


def _canonical(lst: List[int]) -> tuple[int, ...]:
    return tuple(sorted(lst or []))


def _find_duplicate_route(db: Session, station_list: List[int]) -> Route | None:
    n = len(station_list)
    candidates = db.execute(
        select(Route).where(func.cardinality(Route.station_list) == n)
    ).scalars().all()
    target = _canonical(station_list)
    for r in candidates:
        if _canonical(r.station_list) == target:
            return r
    return None


def _get_route_by_name(db: Session, route_name: str) -> Route | None:
    return db.execute(select(Route).where(Route.route_name == route_name)).scalar_one_or_none()


def order_stations_by_nearest(
    db: Session, start_station_id: int, end_station_id: int, station_list: list[int]
) -> list[int]:
    stations = db.execute(
        select(Station.station_id, Station.x, Station.y).where(Station.station_id.in_(station_list))
    ).all()
    coord_map = {sid: (x, y) for sid, x, y in stations}

    to_visit = set(station_list) - {start_station_id, end_station_id}
    ordered = [start_station_id]
    cur = start_station_id

    while to_visit:
        cx, cy = coord_map[cur]
        next_sid, next_dist = None, float("inf")
        for sid in to_visit:
            x, y = coord_map[sid]
            d = sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if d < next_dist:
                next_sid, next_dist = sid, d
        ordered.append(next_sid)
        to_visit.remove(next_sid)
        cur = next_sid

    ordered.append(end_station_id)
    return ordered


@router.post("/", response_model=RouteOut, status_code=status.HTTP_201_CREATED)
def create_virtual_route(payload: RouteCreate, response: Response, db: Session = Depends(get_db)):
    if not payload.station_list:
        raise HTTPException(status_code=400, detail="station_list가 비어 있습니다.")

    existing = set(
        db.execute(
            select(Station.station_id).where(Station.station_id.in_(payload.station_list))
        ).scalars().all()
    )
    missing = [sid for sid in payload.station_list if sid not in existing]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={"message": "station_id가 존재하지 않습니다.", "missing": missing},
        )

    start_id = payload.start_station_id
    end_id = payload.end_station_id
    ordered_list = order_stations_by_nearest(db, start_id, end_id, payload.station_list)
    payload.station_list = ordered_list

    # === 추가: route_name이 이미 존재할 때 station_list 불일치 검사 ===
    if payload.route_name:
        by_name = _get_route_by_name(db, payload.route_name)
        if by_name:
            if _canonical(by_name.station_list) != _canonical(payload.station_list):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "이미 존재하는 route_name의 station_list와 불일치합니다.",
                        "route_name": payload.route_name,
                        "existing_station_list": by_name.station_list,
                        "incoming_station_list": payload.station_list,
                    },
                )
            # route_name 동일 + station_list 동일 → 기존 라우트 반환(중복 생성 방지)
            response.status_code = status.HTTP_200_OK
            return RouteOut(
                route_id=by_name.route_id,
                route_name=by_name.route_name,
                start_station_id=by_name.start_station_id,
                end_station_id=by_name.end_station_id,
                station_list=by_name.station_list,
            )

    # 기존 중복(스테이션 리스트 동일) 라우트가 있으면 반환
    dup = _find_duplicate_route(db, payload.station_list)
    if dup:
        response.status_code = status.HTTP_200_OK
        return RouteOut(
            route_id=dup.route_id,
            route_name=dup.route_name,
            start_station_id=dup.start_station_id,
            end_station_id=dup.end_station_id,
            station_list=dup.station_list,
        )

    # 신규 생성
    for _ in range(5):
        rid = _next_virtual_route_id(db)
        try:
            row = Route(
                route_id=rid,
                route_name=payload.route_name,
                start_station_id=payload.start_station_id,
                end_station_id=payload.end_station_id,
                station_list=payload.station_list,
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return RouteOut(
                route_id=row.route_id,
                route_name=row.route_name,
                start_station_id=row.start_station_id,
                end_station_id=row.end_station_id,
                station_list=row.station_list,
            )
        except IntegrityError:
            db.rollback()
            continue

    raise HTTPException(status_code=409, detail="가상 route_id가 할당되지 않았습니다.")

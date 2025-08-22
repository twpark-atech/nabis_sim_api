# app/api/v1/stations.py
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select, and_, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Station
from app.schemas import StationCreate, StationOut

router = APIRouter()

ID_MIN = 400_000_000
ID_MAX = 500_000_000
RADIUS_M = 10.0 # 10.0m

def _next_virtual_station_id(db: Session) -> int:
    max_id = db.execute(
        select(func.max(Station.station_id)).where(
            and_(Station.station_id >= ID_MIN, Station.station_id < ID_MAX)
        )
    ).scalar()
    if max_id is None:
        return ID_MIN
    nxt = max_id + 1
    if nxt >= ID_MAX:
        raise HTTPException(status_code=409, detail="station_id가 초과되었습니다.")
    return nxt

@router.post("/", response_model=StationOut, status_code=status.HTTP_201_CREATED)
def create_station(payload: StationCreate, response: Response, db: Session = Depends(get_db)):
    lon0 = float(payload.x)
    lat0 = float(payload.y)

    station_geom = func.ST_SetSRID(func.ST_MakePoint(Station.x, Station.y), 4326)
    target_geom = func.ST_SetSRID(func.ST_MakePoint(lon0, lat0), 4326)

    near = db.execute(
        select(Station)
        .where(func.ST_DistanceSphere(station_geom, target_geom) <= RADIUS_M)
        .order_by(func.ST_DistanceSphere(station_geom, target_geom))
        .limit(1)
    ).scalars().first()

    if near:
        response.status_code = status.HTTP_200_OK
        return StationOut(
            station_id=near.station_id,
            station_name=near.station_name,
            x=near.x,
            y=near.y
        )
    
    for _ in range(5):
        sid = _next_virtual_station_id(db)
        try:
            row = Station(
                station_id=sid,
                station_name=payload.station_name,
                x=lon0,
                y=lat0
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return StationOut(
                station_id=row.station_id,
                station_name=row.station_name,
                x=row.x,
                y=row.y
            )
        except IntegrityError:
            db.rollback()
            continue
    
    raise HTTPException(status_code=409, detail="가상 station_id가 할당되지 않았습니다.")
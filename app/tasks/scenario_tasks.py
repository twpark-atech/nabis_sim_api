# app/tasks/scenario_tasks.py
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from celery.utils.log import get_task_logger
from shapely import wkt as _shp_wkt
from shapely.geometry import Point

from app.celery_app import celery_app
from app.db import SessionLocal
from app.models import Scenario, Route
from controllers.Module import compute_total_length  # type: ignore

# === 가져올 유틸/헬퍼들은 "베이스라인" 코드의 이름을 그대로 사용합니다. ===
from app.api.v1.scenarios import (
    search_link_list,
    search_station_map,
    search_avg_speed,
    build_speed_and_positions_0p1s_with_stations,
    compute_route_curvature as _compute_route_curvature,
)

log = get_task_logger(__name__)

# 기본값
DEFAULT_V_MAX = 30.0         # km/h (링크별 속도 부재시 fallback: build 함수의 params.kmh_default로 사용)
TL_STOP_PROB = 1.0 / 3.0     # 신호등 정차 확률(자동 로드한 모든 신호등 대상)
TL_BASE_SEC = 105.0          # 신호등 평균 정차 시간
TL_JITTER_SEC = 70.0         # ±70초

# 좌표계: 링크 보간용 메트릭 SRID(미터), 출력은 WGS84
METRIC_SRID = 5179
OUTPUT_SRID = 4326


def _pick_start_end_points(
    station_list: List[int],
    station_map: List[Tuple[int, str]],
) -> Tuple[Optional[Point], Optional[Point]]:
    """
    station_map: [(station_id, WKT_POINT)]
    station_list의 첫/마지막 정류장 좌표를 shapely Point로 반환
    (search_station_map의 기본 SRID=4326을 그대로 사용)
    """
    if not station_list or not station_map:
        return None, None
    sid2wkt: Dict[int, str] = {sid: w for sid, w in station_map}
    start_pt = _shp_wkt.loads(sid2wkt.get(station_list[0])) if station_list[0] in sid2wkt else None
    end_pt = _shp_wkt.loads(sid2wkt.get(station_list[-1])) if station_list[-1] in sid2wkt else None
    return start_pt, end_pt


@celery_app.task(
    bind=True,
    name="scenarios.generate_and_persist",
    acks_late=True,
    autoretry_for=(OperationalError,),
    retry_backoff=5,
    retry_kwargs={"max_retries": 3},
)
def generate_and_persist(self, payload: Dict[str, Any], scenario_id: int) -> None:
    """
    시나리오(scenario_id)를 읽어 라우팅→속도/좌표 생성→DB 저장까지 처리.
    - 정류장: 항상 정차(기본 60초 대기)
    - 신호등: 경로상의 모든 신호등을 자동 로드하여 TL_STOP_PROB 확률로 정차,
              대기시간 U(TL_BASE_SEC - TL_JITTER_SEC, TL_BASE_SEC + TL_JITTER_SEC)
    - 0.1초 간격 speed_list(km/h)와 coord_list((lon,lat), SRID=4326)을 생성/저장
    """
    db = SessionLocal()
    try:
        sc: Optional[Scenario] = db.get(Scenario, scenario_id)
        if sc is None:
            raise RuntimeError(f"scenario_id={scenario_id} not found")

        # 입력 파라미터
        route_id = int(payload["route_id"])
        path_type = str(payload.get("path_type", "optimal"))

        # 라우트/정류장 로드
        route: Optional[Route] = db.execute(select(Route).where(Route.route_id == route_id)).scalar_one_or_none()
        if not route:
            sc.status = "생성 실패"
            db.commit()
            raise RuntimeError(f"route_id={route_id} not found")

        station_list: List[int] = route.station_list or []
        if len(station_list) < 2:
            sc.status = "생성 실패"
            db.commit()
            raise RuntimeError(f"station_list empty route_id={route_id}")

        # 링크 경로 생성
        link_list: List[str] = search_link_list(db, station_list, path_type)
        if not link_list:
            sc.status = "생성 실패"
            db.commit()
            raise RuntimeError(f"failed to build link path route_id={route_id}")

        # 정류장 좌표 조회 (WKT, 4326)
        station_map = search_station_map(db, station_list)  # [(station_id, WKT)]
        start_pt, end_pt = _pick_start_end_points(station_list, station_map)

        # 링크별 평균속도(km/h) 조회 (5분단위 내림)
        link_speed_dict = search_avg_speed(db, link_list, sc.departure_time)

        # 속도/좌표 생성
        # - station_dwell_seconds: 정류장 정차시간(초), 필요시 payload로 오버라이드
        station_dwell_seconds = float(payload.get("station_dwell_seconds", 60.0))

        speed_list, coord_list = build_speed_and_positions_0p1s_with_stations(
            db=db,
            link_list=link_list,
            link_speed_dict=link_speed_dict,
            station_map=station_map,
            start_point_wgs84=start_pt,
            end_point_wgs84=end_pt,
            stop_tolerance_m=20.0,
            station_dwell_seconds=station_dwell_seconds,  # 정류장 기본 대기
            # ▼ 신호등 자동 로드 + 1/3 확률 정차
            stop_on_tlights=True,
            tlight_stop_nodes=None,                      # 화이트리스트 사용하지 않음(전체 신호등 자동 대상)
            tlight_stop_probability=TL_STOP_PROB,
            tlight_dwell_base=TL_BASE_SEC,
            tlight_dwell_variation=TL_JITTER_SEC,
            metric_srid=METRIC_SRID,                     # 미터 좌표계(예: 5179)
            output_srid=OUTPUT_SRID,                     # 결과는 4326(lon,lat)
            # params=MotionParams(dt=0.1, a_accel=1.5, a_decel=2.0, kmh_default=DEFAULT_V_MAX),  # 필요 시 사용
            random_seed=payload.get("seed"),
            stop_at_route_end=True,
        )

        # 요약값
        route_length_m = float(sum(float(compute_total_length([lid]) or 0.0) for lid in link_list))
        route_curvature = float(_compute_route_curvature(db, link_list))

        # DB 업데이트
        sc.link_list = link_list
        sc.speed_list = speed_list
        sc.coord_list = coord_list
        sc.route_length = route_length_m
        sc.route_curvature = route_curvature
        sc.status = "생성 완료"

        db.commit()
        log.info(
            "Committed scenario_id=%s links=%d speeds=%d coords=%d",
            sc.scenario_id, len(link_list), len(speed_list), len(coord_list)
        )
    except Exception:
        db.rollback()
        try:
            sc2 = db.get(Scenario, scenario_id)
            if sc2:
                sc2.status = "생성 실패"
                db.commit()
        except Exception:
            db.rollback()
        raise
    finally:
        db.close()

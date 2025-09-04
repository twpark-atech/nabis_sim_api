# app/tasks/scenario_tasks.py
from __future__ import annotations
from typing import Dict, Any, Optional, List
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError
from celery.utils.log import get_task_logger

from app.celery_app import celery_app
from app.db import SessionLocal
from app.models import Scenario, Route
from app.api.v1.scenarios import (
    _build_links_and_station_stop_indices,
    _compute_route_curvature,
    _make_speed_and_coords,
    DEFAULT_V_MAX, TL_STOP_PROB, TL_BASE_SEC, TL_JITTER_SEC,
)
from controllers.Module import compute_total_length  # type: ignore

log = get_task_logger(__name__)

@celery_app.task(
    bind=True,
    name="scenarios.generate_and_persist",
    acks_late=True,
    autoretry_for=(OperationalError,),
    retry_backoff=5,
    retry_kwargs={"max_retries": 3},
)
def generate_and_persist(self, payload: Dict[str, Any], scenario_id: int) -> None:
    db = SessionLocal()
    try:
        sc: Optional[Scenario] = db.get(Scenario, scenario_id)
        if sc is None:
            raise RuntimeError(f"scenario_id={scenario_id} not found")

        # 입력값/레퍼런스 시간
        route_id = int(payload["route_id"])
        path_type = str(payload.get("path_type", "optimal"))
        dep_t = sc.departure_time  # DB에 time 타입으로 저장됨
        ref_hour = int(getattr(dep_t, "hour", 15))

        # 라우트/정류장 체크
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

        # 링크 경로 구성
        link_list, stop_idx_to_station_order = _build_links_and_station_stop_indices(db, station_list, path_type)
        if not link_list:
            sc.status = "생성 실패"
            db.commit()
            raise RuntimeError(f"failed to build link path route_id={route_id}")

        # 속도/좌표 생성 (링크별 vmax 커스터마이즈는 이후 단계에서 추가)
        speed_list, coord_list = _make_speed_and_coords(
            db=db,
            link_ids=link_list,
            stop_idx_to_station_order=stop_idx_to_station_order,
            station_list=station_list,
            station_dwell_sec=[60.0] * (len(station_list) - 1),
            v_max_kmh=DEFAULT_V_MAX,
            p_stop_tl=TL_STOP_PROB,
            tl_base_sec=TL_BASE_SEC,
            tl_jitter_sec=TL_JITTER_SEC,
            seed=None,
        )

        # 요약값
        route_length_m = float(sum(float(compute_total_length([lid]) or 0.0) for lid in link_list))
        route_curvature = _compute_route_curvature(db, link_list)

        # 업데이트 + 완료
        sc.link_list = link_list
        sc.speed_list = speed_list
        sc.coord_list = coord_list
        sc.route_length = route_length_m
        sc.route_curvature = route_curvature
        sc.status = "생성 완료"

        db.commit()
        log.info("Committed scenario_id=%s links=%d speeds=%d coords=%d", sc.scenario_id, len(link_list), len(speed_list), len(coord_list))
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

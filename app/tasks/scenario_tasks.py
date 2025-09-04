# app/tasks/scenario_tasks.py
from __future__ import annotations
from typing import Dict, Any, Optional, List
from dateutil import parser as date_parser

from app.celery_app import celery_app
from sqlalchemy import select
from app.db import SessionLocal

from app.models import Scenario, Route
from app.api.v1.scenarios import (
    _build_links_and_station_stop_indices,
    _compute_route_curvature,
    _make_speed_and_coords,
    _fetch_per_link_vmax_from_traffic,
    DEFAULT_V_MAX,
    TL_STOP_PROB,
    TL_BASE_SEC,
    TL_JITTER_SEC,
)
from controllers.Module import compute_total_length


@celery_app.task(name="scenarios.generate_and_persist", acks_late=True)
def generate_and_persist(payload: Dict[str, Any]) -> None:
    db = SessionLocal()
    try:
        route_id = int(payload["route_id"])
        path_type = str(payload.get("path_type", "optimal"))

        route: Optional[Route] = db.execute(
            select(Route).where(Route.route_id == route_id)
        ).scalar_one_or_none()
        if not route:
            return
        
        station_list: List[int] = route.station_list or []
        if len(station_list) < 2:
            return
        
        link_list, stop_idx_to_station_order = _build_links_and_station_stop_indices(
            db, station_list, path_type
        )
        if not link_list:
            return
        
        dep = payload.get("departure_time")
        try:
            if isinstance(dep, str):
                dep_dt = date_parser.isoparse(dep)  # 2025-08-01T15:00:00Z 지원
            else:
                dep_dt = dep  # datetime 객체 허용
            ref_hour = int(getattr(dep_dt, "hour", 15))
        except Exception:
            ref_hour = 15

        per_link_max = _fetch_per_link_vmax_from_traffic(db, link_list, ref_hour=ref_hour)

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
            seed=42,
            per_link_vmax=per_link_max,
        )

        route_length_m = float(
            sum(float(compute_total_length([lid]) or 0.0) for lid in link_list)
        )
        route_curvature = _compute_route_curvature(db, link_list)

        scenario = Scenario(
            name=str(payload["name"]),
            route_id=route_id,
            headway_min=int(payload["headway_min"]),
            start_time=payload["start_time"],
            end_time=payload["end_time"],
            departure_time=payload["departure_time"],
            path_type=path_type,
            route_length=route_length_m,
            route_curvature=route_curvature,
            speed_list=speed_list,
            coord_list=coord_list,
            link_list=link_list,
        )
        db.add(scenario)
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close() 
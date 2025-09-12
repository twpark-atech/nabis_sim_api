# app/api/v1/scenarios.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Dict, Optional, Tuple, Iterable
from dataclasses import dataclass
from datetime import datetime, time as _time
import math
import random

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text, select
from shapely import wkt as _shp_wkt
from shapely.geometry import LineString, Point
from shapely.ops import substring
import numpy as np
from pyproj import Transformer

from app.db import get_db
from app.models import Route
from app.api.v1.paths import compute_links_via_router
from app.schemas import ScenarioCreate, ScenarioOut
from controllers.Module import compute_total_length  # noqa: F401

router = APIRouter()

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
EPS_RATIO = 1e-4  # 좌표 보간 시 0/1 경계 달라붙음 방지용

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _round_to_5min_floor(t: str | datetime | _time) -> str:
    if isinstance(t, datetime):
        hh, mm, _ = t.hour, t.minute, t.second
    elif isinstance(t, _time):
        hh, mm, _ = t.hour, t.minute, t.second
    else:
        dt = datetime.strptime(str(t), "%H:%M:%S")
        hh, mm, _ = dt.hour, dt.minute, dt.second
    mm = (mm // 5) * 5
    return f"{hh:02d}:{mm:02d}:00"


def _polyline_turn_sum_radians(line: Optional[LineString]) -> float:
    if not line:
        return 0.0
    coords = list(line.coords)
    if len(coords) < 3:
        return 0.0

    def heading(p0, p1):
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        return np.arctan2(dy, dx)

    total = 0.0
    prev_h = heading(coords[0], coords[1])
    for i in range(1, len(coords) - 1):
        h = heading(coords[i], coords[i + 1])
        d = np.arctan2(np.sin(h - prev_h), np.cos(h - prev_h))
        total += abs(d)
        prev_h = h
    return float(total)


# -----------------------------------------------------------------------------
# DB Helpers
# -----------------------------------------------------------------------------
def search_station_map(db: Session, station_list: List[int], srid: int = 4326) -> List[Tuple[int, str]]:
    if not station_list:
        return []
    q = text("""
        SELECT s."station_id"::bigint AS station_id,
               ST_AsText(ST_SetSRID(ST_MakePoint(s.x, s.y), :srid)) AS wkt
        FROM "SIM_BIS_BUS_STATION_LOCATION" AS s
        WHERE s."station_id"::bigint = ANY(:station_ids)
    """)
    rows = db.execute(q, {"station_ids": station_list, "srid": srid}).fetchall()
    return [(int(r[0]), str(r[1])) for r in rows if r and r[1]]


def search_link_list(db: Session, station_list: List[int], path_type: str) -> List[str]:
    if not station_list or len(station_list) < 2:
        return []
    full_links: List[str] = []
    for i in range(len(station_list) - 1):
        seg_links = compute_links_via_router(
            db,
            start_station_id=station_list[i],
            end_station_id=station_list[i + 1],
            ptype=path_type,
        ) or []
        if not seg_links:
            continue
        if full_links and full_links[-1] == seg_links[0]:
            seg_links = seg_links[1:]
        full_links.extend(seg_links)
    return full_links


def _as_text_array(ids: Iterable[object]) -> List[str]:
    """바인딩용으로 LINK_ID 리스트를 TEXT[]로 변환."""
    return [str(x) for x in ids if x is not None]


def search_traffic_map(db: Session, link_list: List[object]) -> List[Tuple[str, str, str]]:
    if not link_list:
        return []
    link_ids_txt = _as_text_array(link_list)
    q = text("""
        SELECT nu."LINK_ID"::text AS lid,
               tl."NODE_ID"::text AS node_id,
               ST_AsText(tl.geometry) AS wkt
        FROM new_uroad AS nu
        JOIN utraffic_light_info AS tl
          ON tl."NODE_ID"::text = ANY(ARRAY[nu."F_NODE"::text, nu."T_NODE"::text])
        WHERE nu."LINK_ID"::text = ANY(:link_ids)
    """)
    rows = db.execute(q, {"link_ids": link_ids_txt}).fetchall()
    return [(str(r[0]), str(r[1]), str(r[2])) for r in rows]


def _get_links_geometry_wkt_wgs84(db: Session, link_ids: List[object]) -> Dict[str, str]:
    if not link_ids:
        return {}
    link_ids_txt = _as_text_array(link_ids)
    q = text("""
        SELECT nu."LINK_ID"::text AS lid,
               ST_AsText(ST_Transform(nu.geometry, 4326)) AS wkt_geom
        FROM new_uroad AS nu
        WHERE nu."LINK_ID"::text = ANY(:link_ids)
    """)
    rows = db.execute(q, {"link_ids": link_ids_txt}).fetchall()
    return {str(r[0]): str(r[1]) for r in rows if r and r[1]}


def compute_route_curvature(db: Session, link_list: List[str]) -> float:
    if not link_list:
        return 0.0
    total_m = float(compute_total_length(link_list) or 0.0)
    if total_m <= 0:
        return 0.0
    total_km = total_m / 1000.0
    wkt_map = _get_links_geometry_wkt_wgs84(db, link_list)
    turn_sum = 0.0
    for lid in link_list:
        wkt = wkt_map.get(lid)
        if not wkt:
            continue
        try:
            line = _shp_wkt.loads(wkt)
        except Exception:
            continue
        turn_sum += _polyline_turn_sum_radians(line)
    return round(turn_sum / total_km, 6)


def search_avg_speed(db: Session, link_list: List[object], depart_time: str | datetime | _time) -> Dict[str, float]:
    if not link_list:
        return {}
    slot = _round_to_5min_floor(depart_time)
    link_ids_txt = _as_text_array(link_list)
    q = text("""
        SELECT nu."LINK_ID"::text AS lid,
               stc.avg_speed
        FROM new_uroad AS nu
        JOIN sim_traffic_congest AS stc
          ON stc.roadname::text = nu."ROAD_NAME"
        WHERE nu."LINK_ID"::text = ANY(:link_ids)
          AND stc.slot_5min::text = :slot
    """)
    rows = db.execute(q, {"link_ids": link_ids_txt, "slot": slot}).fetchall()
    out: Dict[str, float] = {}
    for lid, avg_spd in rows:
        if lid is None or avg_spd is None:
            continue
        try:
            val = float(avg_spd)
        except Exception:
            continue
        if val > 0.0:
            out[str(lid)] = val
    return out


# -----------------------------------------------------------------------------
# Speed/Geom helpers
# -----------------------------------------------------------------------------
def _get_links_linestrings_metric(
    db: Session,
    link_ids: List[object],
    srid_metric: int = 5179,  # 미터 좌표계
) -> Dict[str, LineString]:
    if not link_ids:
        return {}
    link_ids_txt = _as_text_array(link_ids)
    q = text("""
        SELECT nu."LINK_ID"::text AS lid,
               ST_AsText(ST_Transform(nu.geometry, :srid)) AS wkt_geom
        FROM new_uroad AS nu
        WHERE nu."LINK_ID"::text = ANY(:link_ids)
    """)
    rows = db.execute(q, {"link_ids": link_ids_txt, "srid": srid_metric}).fetchall()
    out: Dict[str, LineString] = {}
    for lid, wkt in rows:
        if wkt:
            try:
                out[str(lid)] = _shp_wkt.loads(wkt)
            except Exception:
                continue
    return out


def _transform_point(pt: Point, src_epsg: int = 4326, dst_epsg: int = 4326) -> Point:
    transformer = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    x, y = transformer.transform(pt.x, pt.y)
    return Point(x, y)


def _clip_line_by_points(
    line: LineString,
    start_point: Optional[Point] = None,
    end_point: Optional[Point] = None,
) -> LineString:
    if line is None or line.is_empty:
        return line
    start_d = 0.0
    end_d = line.length
    if start_point is not None:
        start_d = float(line.project(start_point))
    if end_point is not None:
        end_d = float(line.project(end_point))
    if start_d > end_d:
        start_d, end_d = end_d, start_d
    return substring(line, start_d, end_d)


@dataclass
class MotionParams:
    dt: float = 0.1
    a_accel: float = 1.5
    a_decel: float = 2.0
    kmh_default: float = 30.0


def _kmh_to_mps(v_kmh: float) -> float:
    return float(v_kmh) * (1000.0 / 3600.0)


def _distance_for_accel(v0: float, v1: float, a: float) -> float:
    if a == 0:
        return 0.0
    return (v1 * v1 - v0 * v0) / (2.0 * a)


def _solve_vpeak_for_short_segment(L: float, v0: float, v_end: float, a_accel: float, a_decel: float) -> float:
    A = 0.5 / a_accel + 0.5 / a_decel
    B = L + (v0 * v0) / (2.0 * a_accel) + (v_end * v_end) / (2.0 * a_decel)
    vpeak_sq = max(B / A, 0.0)
    return math.sqrt(vpeak_sq)


# -----------------------------------------------------------------------------
# Orientation helpers (pairs & lines)
# -----------------------------------------------------------------------------
def _reverse_line(line: LineString) -> LineString:
    return LineString(list(line.coords)[::-1])


def _endpoint_distance(a: LineString, b: LineString) -> Tuple[float, float, float, float]:
    a0 = Point(a.coords[0]); a1 = Point(a.coords[-1])
    b0 = Point(b.coords[0]); b1 = Point(b.coords[-1])
    return (a0.distance(b0), a0.distance(b1), a1.distance(b0), a1.distance(b1))


def _orient_link_pairs_by_connectivity(pairs: List[Tuple[str, LineString]], tol: float = 1.0) -> List[Tuple[str, LineString]]:
    """(link_id, LineString) 쌍을 방향 정렬."""
    if not pairs:
        return []
    out: List[Tuple[str, LineString]] = [pairs[0]]
    for i in range(1, len(pairs)):
        _, prev_ln = out[-1]
        lid, cur_ln = pairs[i]
        d00, d01, d10, d11 = _endpoint_distance(prev_ln, cur_ln)
        if d10 <= d11:
            out.append((lid, cur_ln))
        else:
            out.append((lid, _reverse_line(cur_ln)))
    return out


def _orient_lines_by_connectivity(lines: List[LineString], tol: float = 1.0) -> List[LineString]:
    """LineString 리스트를 방향 정렬."""
    if not lines:
        return []
    oriented: List[LineString] = [lines[0]]
    for i in range(1, len(lines)):
        prev = oriented[-1]
        cur = lines[i]
        d00, d01, d10, d11 = _endpoint_distance(prev, cur)
        if d10 <= d11:
            oriented.append(cur)
        else:
            oriented.append(_reverse_line(cur))
    return oriented


# -----------------------------------------------------------------------------
# Segmenting and profiles
# -----------------------------------------------------------------------------
def _station_points_wgs84(station_map: List[Tuple[int, str]]) -> List[Point]:
    pts = []
    for _, wkt in station_map:
        try:
            pts.append(_shp_wkt.loads(wkt))
        except Exception:
            continue
    return pts


def _traffic_points_wgs84(
    traffic_map: Optional[List[Tuple[str, str, str]]],
    only_node_ids: Optional[Iterable[object]] = None,
) -> List[Point]:
    """
    traffic_map: (link_id, node_id, wkt)
    only_node_ids 가 주어지면 그 node_id만 필터링하여 포인트 생성.
    """
    pts = []
    if not traffic_map:
        return pts
    allow: Optional[set[str]] = None
    if only_node_ids is not None:
        allow = {str(x) for x in only_node_ids}
    for _, node_id, wkt in traffic_map:
        if allow is not None and str(node_id) not in allow:
            continue
        try:
            pts.append(_shp_wkt.loads(wkt))
        except Exception:
            continue
    return pts


def _collect_stops_with_dwell(
    pairs_metric: List[Tuple[str, LineString]],
    station_pts_wgs84: List[Point],
    tlight_pts_wgs84: List[Point],
    *,
    metric_srid: int,
    tol_m: float,
    station_dwell_s: float,
    tlight_stop_prob: float,
    tlight_dwell_base: float,
    tlight_dwell_var: float,
    rng: random.Random,
) -> Dict[int, List[Tuple[float, float]]]:
    if not pairs_metric:
        return {}

    to_metric = Transformer.from_crs(4326, metric_srid, always_xy=True).transform
    station_pts_metric = [Point(*to_metric(p.x, p.y)) for p in station_pts_wgs84] if station_pts_wgs84 else []
    tlight_pts_metric = [Point(*to_metric(p.x, p.y)) for p in tlight_pts_wgs84] if tlight_pts_wgs84 else []

    out: Dict[int, List[Tuple[float, float]]] = {}

    for idx, (_lid, line) in enumerate(pairs_metric):
        if line is None or line.is_empty or line.length <= 0:
            continue

        # 링크 길이에 비례한 작은 ε (끝점 달라붙음 방지)
        eps = max(0.5, min(line.length * 1e-3, tol_m * 0.5))

        dist_dwell: List[Tuple[float, float]] = []

        # --- 정류장: 항상 정차 ---
        for pt in station_pts_metric:
            if pt.distance(line) <= tol_m:
                d = float(line.project(pt))
                # 끝점이면 ε만큼 내부로 클램프
                d = float(np.clip(d, eps, line.length - eps))
                dist_dwell.append((d, max(0.0, station_dwell_s)))

        # --- 신호등: 확률적 정차 ---
        if tlight_pts_metric and tlight_stop_prob > 0.0:
            for pt in tlight_pts_metric:
                if pt.distance(line) <= tol_m:
                    if rng.random() <= tlight_stop_prob:
                        d = float(line.project(pt))
                        d = float(np.clip(d, eps, line.length - eps))
                        dwell = rng.uniform(tlight_dwell_base - tlight_dwell_var,
                                            tlight_dwell_base + tlight_dwell_var)
                        dist_dwell.append((d, max(0.0, dwell)))

        # 근접 지점 병합
        if dist_dwell:
            dist_dwell.sort(key=lambda x: x[0])
            merged: List[Tuple[float, float]] = []
            for d, dw in dist_dwell:
                if not merged:
                    merged.append((d, dw)); continue
                d_prev, dw_prev = merged[-1]
                if abs(d - d_prev) <= tol_m:
                    merged[-1] = (0.5 * (d + d_prev), max(dw_prev, dw))
                else:
                    merged.append((d, dw))
            out[idx] = merged

    return out


def _split_line_by_stops(line: LineString, stops: List[Tuple[float, float]]) -> List[Tuple[LineString, float]]:
    """line을 정지 지점(stops: [(dist, dwell_s)])으로 분할."""
    if not stops:
        return [(line, 0.0)]
    out: List[Tuple[LineString, float]] = []
    cur = 0.0
    for d, dwell in stops:
        seg = substring(line, cur, d)
        if seg and seg.length > 0:
            out.append((seg, float(dwell)))
        cur = d
    tail = substring(line, cur, line.length)
    if tail and tail.length > 0:
        out.append((tail, 0.0))
    return out


def _segment_speeds_mps(
    L: float,
    v0: float,
    vmax: float,
    v_end: float,
    dt: float,
    a_acc: float,
    a_dec: float,
) -> Tuple[List[float], float]:
    def d_for(vs, ve, a):
        return max(_distance_for_accel(vs, ve, a), 0.0)

    d_to_vmax = d_for(v0, vmax, a_acc)
    d_from_vmax = d_for(vmax, v_end, -a_dec)
    if L >= d_to_vmax + d_from_vmax:
        v_peak = vmax
        d_accel, d_cruise, d_decel = d_to_vmax, L - (d_to_vmax + d_from_vmax), d_from_vmax
    else:
        v_peak = min(_solve_vpeak_for_short_segment(L, v0, v_end, a_acc, a_dec), vmax)
        d_accel, d_cruise, d_decel = d_for(v0, v_peak, a_acc), 0.0, d_for(v_peak, v_end, -a_dec)

    seq: List[float] = []
    s = 0.0; v = v0
    while s < d_accel - 1e-6:
        v_next = min(v + a_acc * dt, v_peak)
        ds = (v + v_next) * 0.5 * dt
        if s + ds > d_accel:
            remaining = d_accel - s
            avg_v = max((v + v_next) * 0.5, 1e-6)
            dt_scaled = remaining / avg_v
            v_next = min(v + a_acc * dt_scaled, v_peak)
            seq.append(v_next); v = v_next; s = d_accel; break
        seq.append(v_next); v = v_next; s += ds

    s = 0.0
    if d_cruise > 1e-6:
        while s < d_cruise - 1e-6:
            v_next = v_peak
            ds = v_next * dt
            if s + ds > d_cruise:
                seq.append(v_next); s = d_cruise; v = v_next; break
            seq.append(v_next); s += ds; v = v_next

    s = 0.0
    if d_decel > 1e-6:
        while s < d_decel - 1e-6:
            v_next = max(v - a_dec * dt, v_end)
            ds = (v + v_next) * 0.5 * dt
            if s + ds > d_decel:
                remaining = d_decel - s
                avg_v = max((v + v_next) * 0.5, 1e-6)
                dt_scaled = remaining / avg_v
                v_next = max(v - a_dec * dt_scaled, v_end)
                seq.append(v_next); v = v_next; s = d_decel; break
            seq.append(v_next); v = v_next; s += ds
    return seq, v


# -----------------------------------------------------------------------------
# 예전 방식 참고: 좌표 재구성 유틸 (speed → 누적거리 → 링크 보간)
# -----------------------------------------------------------------------------
def _fetch_link_nodes_text(db: Session, link_ids: List[object]) -> Dict[str, Tuple[Optional[int], Optional[int]]]:
    """LINK_ID(text) -> (F_NODE, T_NODE)"""
    lids_text = _as_text_array(link_ids)
    if not lids_text:
        return {}
    q = text("""
        SELECT "LINK_ID"::text AS lid_text, "F_NODE", "T_NODE"
        FROM new_uroad
        WHERE "LINK_ID"::text = ANY(:lid_list)
    """)
    rows = db.execute(q, {"lid_list": lids_text}).fetchall()
    out: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
    for lid_text, fnode, tnode in rows:
        out[str(lid_text)] = (fnode, tnode)
    return out


def _approx_dist_m(a_xy: Tuple[float, float], b_xy: Tuple[float, float]) -> float:
    """경위도(4326) 두 점 사이 대략 거리(m)"""
    ax, ay = a_xy
    bx, by = b_xy
    dx = (ax - bx) * np.cos(np.deg2rad((ay + by) * 0.5))
    dy = (ay - by)
    return float(np.hypot(dx, dy) * 111_320.0)


def _orient_links_by_intersection_text(db: Session, link_ids: List[object]) -> List[Tuple[str, bool]]:
    """
    링크 진행 방향 결정 (노드 교집합/근사거리 기반).
    반환: [(LINK_ID(text), flip_bool)]
    flip=True 면 geometry를 뒤집어서 사용.
    """
    lids_txt = _as_text_array(link_ids)
    n = len(lids_txt)
    if n == 0:
        return []

    nodes = _fetch_link_nodes_text(db, lids_txt)  # lid_text -> (F_NODE, T_NODE)
    wkt_map = _get_links_geometry_wkt_wgs84(db, link_ids)  # lid_text -> WKT
    geoms: Dict[str, Optional[LineString]] = {}
    for lid in lids_txt:
        wkt = wkt_map.get(lid)
        try:
            geoms[lid] = _shp_wkt.loads(wkt) if wkt else None
        except Exception:
            geoms[lid] = None

    def choose_flip_for_end(lid_text: str, end_target: Optional[int]) -> bool:
        f, t = nodes.get(lid_text, (None, None))
        if end_target is None:
            return False
        if t == end_target:
            return False
        if f == end_target:
            return True
        return False

    def choose_flip_for_start(lid_text: str, start_target: Optional[int]) -> bool:
        f, t = nodes.get(lid_text, (None, None))
        if start_target is None:
            return False
        if f == start_target:
            return False
        if t == start_target:
            return True
        return False

    flips: List[Optional[bool]] = [None] * n
    i = 0
    while i < n:
        if i == n - 1:
            flips[i] = False if flips[i] is None else flips[i]
            break

        lid_i, lid_j = lids_txt[i], lids_txt[i + 1]
        fi, ti = nodes.get(lid_i, (None, None))
        fj, tj = nodes.get(lid_j, (None, None))

        # 1) 인접 교집합으로 우선 결정
        common_ij = ({fi, ti}.intersection({fj, tj}) - {None})
        if common_ij:
            s = next(iter(common_ij))
            if flips[i] is None:
                flips[i] = choose_flip_for_end(lid_i, s)
            if flips[i + 1] is None:
                flips[i + 1] = choose_flip_for_start(lid_j, s)
            i += 1
            continue

        # 2) lookahead(다다음 링크)로 결정 보조
        if i + 2 < n:
            lid_k = lids_txt[i + 2]
            fk, tk = nodes.get(lid_k, (None, None))
            common_jk = ({fj, tj}.intersection({fk, tk}) - {None})
            if common_jk:
                s12 = next(iter(common_jk))
                flips[i + 1] = choose_flip_for_end(lid_j, s12)

                gj = geoms.get(lid_j)
                start_j = tj if flips[i + 1] else fj

                if flips[i] is None:
                    if start_j in {fi, ti}:
                        flips[i] = choose_flip_for_end(lid_i, start_j)
                    else:
                        gi = geoms.get(lid_i)
                        if gi is not None and gj is not None and len(gi.coords) >= 2 and len(gj.coords) >= 2:
                            start_j_xy = (gj.coords[-1][0], gj.coords[-1][1]) if flips[i + 1] else (gj.coords[0][0], gj.coords[0][1])
                            end_i_xy_false = (gi.coords[-1][0], gi.coords[-1][1])
                            end_i_xy_true = (gi.coords[0][0], gi.coords[0][1])
                            flips[i] = (_approx_dist_m(end_i_xy_true, start_j_xy) <
                                        _approx_dist_m(end_i_xy_false, start_j_xy))
                        else:
                            flips[i] = False
                i += 1
                continue

        # 3) 기하 근사로 결정
        if flips[i] is None:
            gi = geoms.get(lid_i)
            gj = geoms.get(lid_j)
            if gi is not None and gj is not None and len(gi.coords) >= 2 and len(gj.coords) >= 2:
                start_j_xy = (gj.coords[0][0], gj.coords[0][1])
                end_i_xy_false = (gi.coords[-1][0], gi.coords[-1][1])
                end_i_xy_true = (gi.coords[0][0], gi.coords[0][1])
                flips[i] = (_approx_dist_m(end_i_xy_true, start_j_xy) <
                            _approx_dist_m(end_i_xy_false, start_j_xy))
            else:
                flips[i] = False

        if flips[i + 1] is None:
            flips[i + 1] = False
        i += 1

    flips = [bool(f) if f is not None else False for f in flips]
    return list(zip(lids_txt, flips))


def _cum_dists_from_speeds_kmh_for_coords(speeds: List[float], dt: float) -> np.ndarray:
    """속도(km/h) 시퀀스 -> 누적이동거리(m), trapezoidal"""
    if not speeds:
        return np.array([], dtype=float)
    v = np.array(speeds, dtype=float) / 3.6  # m/s
    if len(v) == 1:
        return np.array([0.0], dtype=float)
    step = (v[:-1] + v[1:]) * 0.5 * dt
    cum = np.concatenate([[0.0], np.cumsum(step)])
    return cum


def rebuild_coords_from_speed_list(
    db: Session,
    link_ids: List[object],
    speed_list: List[float],
    dt: float = 0.1,  # 0.1s
) -> List[Tuple[float, float]]:
    """
    좌표 재구성:
    - 링크를 실제 진행 방향으로 정렬/뒤집기
    - 링크별 실제 길이(m) 누적 -> 누적이동거리와 매칭
    - 각 위치를 해당 링크의 normalized ratio로 보간해 좌표 생성
    """
    if not link_ids or not speed_list:
        return []

    # 1) 방향 정렬(노드/기하 기반)
    oriented = _orient_links_by_intersection_text(db, link_ids)  # [(lid_text, flip)]

    # 2) 링크 geometry(4326)와 실제 길이(m)
    lids_txt = [lid for lid, _ in oriented]
    wkt_map = _get_links_geometry_wkt_wgs84(db, lids_txt)
    geoms: List[LineString] = []
    lens_m: List[float] = []
    for lid_text, flip in oriented:
        wkt = wkt_map.get(lid_text)
        if not wkt:
            continue
        try:
            g = _shp_wkt.loads(wkt)
        except Exception:
            continue
        if g is None or len(g.coords) < 2:
            continue
        if flip:
            g = LineString(list(g.coords)[::-1])
        L = float(compute_total_length([lid_text]) or 0.0)  # 실제 m 길이
        if L <= 0.0:
            continue
        geoms.append(g)
        lens_m.append(L)
    if not geoms:
        return [(0.0, 0.0)] * len(speed_list)

    cum_link_m = np.cumsum(lens_m)  # 각 링크 끝까지 누적(m)
    total_path_m = float(cum_link_m[-1])

    # 3) 속도 -> 누적거리(m), 총길이 초과분 클램프
    cum_m = _cum_dists_from_speeds_kmh_for_coords(speed_list, dt=dt)
    cum_m = np.clip(cum_m, 0.0, total_path_m)

    # 4) 각 누적거리 위치를 (어느 링크, ratio)로 변환 후 보간
    coords: List[Tuple[float, float]] = []
    li = 0  # 현재 링크 인덱스
    for d in cum_m:
        while li < len(cum_link_m) - 1 and d > cum_link_m[li] + 1e-9:
            li += 1
        link_start_m = float(cum_link_m[li - 1]) if li > 0 else 0.0
        link_len_m = float(lens_m[li])
        inside_m = float(d - link_start_m)
        r = inside_m / max(1e-9, link_len_m)
        r = float(np.clip(r, 0.0 + EPS_RATIO, 1.0 - EPS_RATIO))
        pt = geoms[li].interpolate(r, normalized=True)
        coords.append((float(pt.x), float(pt.y)))

    return coords


# -----------------------------------------------------------------------------
# 0.1s speed (km/h) & positions with stops (stations + configurable TL only)
# -----------------------------------------------------------------------------
def build_speed_and_positions_0p1s_with_stations(
    db: Session,
    link_list: List[str],
    link_speed_dict: Dict[str, float],
    station_map: List[Tuple[int, str]],
    *,
    start_point_wgs84: Optional[Point] = None,
    end_point_wgs84: Optional[Point] = None,
    stop_tolerance_m: float = 5.0,
    station_dwell_seconds: float = 60.0,           # 정류장에서는 기본 60초 정차
    stop_on_tlights: bool = False,                 # 기본: 신호등에서는 멈추지 않음
    tlight_stop_nodes: Optional[List[object]] = None,  # 멈출 "설정된" 신호등 NODE_ID 목록
    tlight_stop_probability: float = 1.0,          # whitelist 사용 시 1.0로 적용 권장
    tlight_dwell_base: float = 105.0,
    tlight_dwell_variation: float = 70.0,
    metric_srid: int = 5179,
    output_srid: int = 4326,
    params: MotionParams = MotionParams(),
    random_seed: Optional[int] = None,
    stop_at_route_end: bool = True,
) -> Tuple[List[float], List[Tuple[float, float]]]:

    if not link_list:
        return [], []

    rng = random.Random(random_seed)

    # 링크 + 절단
    link_geoms = _get_links_linestrings_metric(db, link_list, srid_metric=metric_srid)
    sp_metric = _transform_point(start_point_wgs84, 4326, metric_srid) if start_point_wgs84 else None
    ep_metric = _transform_point(end_point_wgs84, 4326, metric_srid) if end_point_wgs84 else None

    pairs: List[Tuple[str, LineString]] = []
    for idx, lid in enumerate(link_list):
        ln = link_geoms.get(lid)
        if ln is None or ln.is_empty:
            continue
        if idx == 0 and sp_metric is not None:
            ln = _clip_line_by_points(ln, start_point=sp_metric, end_point=None)
        if idx == len(link_list) - 1 and ep_metric is not None:
            ln = _clip_line_by_points(ln, start_point=None, end_point=ep_metric)
        if ln and ln.length > 0:
            pairs.append((lid, ln))
    if not pairs:
        return [], []

    # 방향 정렬 (쌍 버전)
    pairs = _orient_link_pairs_by_connectivity(pairs)

    # 정차 지점 수집
    station_pts = _station_points_wgs84(station_map)

    # 신호등 포인트: 기본은 멈추지 않음. whitelist가 오면 그 노드만 고려.
    tlight_pts: List[Point] = []
    traffic_map_raw = None
    if stop_on_tlights or (tlight_stop_nodes is not None and len(tlight_stop_nodes) > 0):
        traffic_map_raw = search_traffic_map(db, link_list)
        tlight_pts = _traffic_points_wgs84(traffic_map_raw, only_node_ids=tlight_stop_nodes)

    # whitelist가 있으면 확률 1.0로 강제(설정한 신호등에서만, 반드시 정차)
    tl_prob = 1.0 if (tlight_stop_nodes is not None and len(tlight_stop_nodes) > 0) else (tlight_stop_probability if stop_on_tlights else 0.0)

    stops_by_link = _collect_stops_with_dwell(
        pairs, station_pts, tlight_pts,
        metric_srid=metric_srid,
        tol_m=stop_tolerance_m,
        station_dwell_s=station_dwell_seconds,
        tlight_stop_prob=tl_prob,
        tlight_dwell_base=tlight_dwell_base,
        tlight_dwell_var=tlight_dwell_variation,
        rng=rng
    )

    # 분할
    segments: List[Tuple[str, LineString, float]] = []
    for idx, (lid, ln) in enumerate(pairs):
        stops = stops_by_link.get(idx, [])
        parts = _split_line_by_stops(ln, stops)
        for seg, dwell_s in parts:
            if seg and seg.length > 0:
                segments.append((lid, seg, dwell_s))

    # 속도 프로파일(m/s)
    v_curr = 0.0
    speeds_mps: List[float] = []
    dt = params.dt
    vmax_map: Dict[str, float] = {}
    for lid, _ in pairs:
        vmax_map[lid] = _kmh_to_mps(link_speed_dict.get(lid, params.kmh_default))

    for i, (lid, seg, dwell_s) in enumerate(segments):
        vmax = vmax_map.get(lid, _kmh_to_mps(params.kmh_default))

        # ⬇️ 멈추는 곳: 정류장/whitelist 신호등/종점만
        is_last = (i == len(segments) - 1)
        if dwell_s > 0.0:
            v_end = 0.0
        elif not is_last:
            next_lid = segments[i + 1][0]
            next_vmax = vmax_map.get(next_lid, _kmh_to_mps(params.kmh_default))
            # 링크 연결 시 “급변 금지”: 다음 링크의 vmax를 고려해 연속적인 종료속도 설정
            # - 다음 vmax가 더 낮으면 미리 감속하여 v_end를 next_vmax로 맞춤
            # - 더 높으면 현재 vmax 유지(다음 링크에서 가속)
            v_end = min(vmax, next_vmax)
        else:
            v_end = 0.0 if stop_at_route_end else vmax

        seq, v_last = _segment_speeds_mps(
            L=float(seg.length),
            v0=v_curr, vmax=vmax, v_end=v_end,
            dt=dt, a_acc=params.a_accel, a_dec=params.a_decel,
        )
        speeds_mps.extend(seq)
        v_curr = v_last

        # 정차 대기 시간 삽입(정류장/신호등)
        if dwell_s > 0.0:
            dwell_steps = int(round(dwell_s / dt))
            if dwell_steps > 0:
                speeds_mps.extend([0.0] * dwell_steps)
                v_curr = 0.0

    # 급가/급감속 리미터 (사후 안정화) — stop&go 억제
    speeds_mps = _enforce_speed_limits(
        speeds_mps,
        dt,
        params.a_accel,
        params.a_decel
    )

    speeds_kmh = [round(v * 3.6, 2) for v in speeds_mps]

    # 좌표는 누적거리 기반으로 링크 위에서만 보간
    coords = rebuild_coords_from_speed_list(
        db=db,
        link_ids=link_list,     # text LINK_ID 리스트
        speed_list=speeds_kmh,  # km/h
        dt=dt,                  # 0.1s
    )
    return speeds_kmh, coords


def _enforce_speed_limits(
    speeds_mps: List[float],
    dt: float,
    a_acc: float,
    a_dec: float,
    iters: int = 2,
) -> List[float]:
    """가감속 한계(dv<=a*dt)를 강제해 급락/급상승 제거."""
    if not speeds_mps:
        return speeds_mps
    sp = speeds_mps[:]  # copy

    for _ in range(iters):
        # (a) 역방향: 감속 제한
        for i in range(len(sp) - 2, -1, -1):
            max_vi = max(sp[i + 1] + a_dec * dt, 0.0)
            if sp[i] > max_vi:
                sp[i] = max_vi

        # (b) 정방향: 가속 제한
        for i in range(len(sp) - 1):
            max_vnext = max(sp[i] + a_acc * dt, 0.0)
            if sp[i + 1] > max_vnext:
                sp[i + 1] = max_vnext

    # 음수 클램프
    for i in range(len(sp)):
        if sp[i] < 0:
            sp[i] = 0.0
    return sp


# -----------------------------------------------------------------------------
# API (옵션)
# -----------------------------------------------------------------------------
@router.post("/", response_model=ScenarioOut, status_code=status.HTTP_201_CREATED)
def create_scenario(payload: ScenarioCreate, db: Session = Depends(get_db)):
    route = db.execute(
        select(Route).where(Route.route_id == payload.route_id)
    ).scalar_one_or_none()
    if route is None:
        raise HTTPException(status_code=404, detail="해당 route_id가 존재하지 않습니다.")

    if payload.path_type not in ("shortest", "optimal"):
        raise HTTPException(status_code=400, detail="path_type은 shortest/optimal 중 하나여야 합니다.")

    station_list: List[int] = list(route.station_list or [])
    if len(station_list) < 2:
        raise HTTPException(status_code=400, detail="해당 노선의 station이 2개 미만입니다.")

    station_map = search_station_map(db, station_list)
    link_list = search_link_list(db, station_list, payload.path_type)
    if not link_list:
        raise HTTPException(status_code=400, detail="유효한 링크 경로를 생성하지 못했습니다.")

    traffic_map = search_traffic_map(db, link_list)
    route_length_m = float(compute_total_length(link_list) or 0.0)
    route_curvature = compute_route_curvature(db, link_list)
    link_speed_dict = search_avg_speed(db, link_list, payload.departure_time)

    # 필요 시 아래 주석 해제하여 속도/좌표 생성 사용
    speeds_kmh, coords = build_speed_and_positions_0p1s_with_stations(
        db=db,
        link_list=link_list,
        link_speed_dict=link_speed_dict,
        station_map=station_map,
        station_dwell_seconds=60.0,            # 정류장 정차
        stop_on_tlights=True,                 # 기본은 신호등 정차 없음
        tlight_stop_nodes=[],                  # 설정한 신호등 NODE_ID 목록 전달 시 여기에 지정
        tlight_stop_probability=0.6,
        tlight_dwell_base=105.0,
        tlight_dwell_variation=70.0,
        metric_srid=5179,
        output_srid=4326,
        params=MotionParams(),
        random_seed=None,
        stop_at_route_end=True,
    )

    return ScenarioOut(
        route_id=payload.route_id,
        path_type=payload.path_type,
        station_map=station_map,
        link_list=link_list,
        traffic_map=traffic_map,
        route_length_m=route_length_m,
        route_curvature=route_curvature,
        link_speed_map=link_speed_dict,
        departure_time=_round_to_5min_floor(payload.departure_time)
    )

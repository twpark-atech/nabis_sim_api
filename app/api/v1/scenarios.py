from __future__ import annotations

from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta, time as _time, date as _date

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, text
from sqlalchemy.orm import Session
from shapely.geometry import LineString
from shapely import wkt as _shp_wkt
import numpy as np

from app.db import get_db
from app.models import Scenario, Route
from app.api.v1.paths import compute_links_via_router
from app.schemas import ScenarioCreate, ScenarioOut

# 이미 보유한 유틸(길이/신호등) 사용
from controllers.Module import compute_total_length, get_nodes_with_traffic_light  # type: ignore

# --------------------------------
# 상수
# --------------------------------
DT = 0.1  # 100ms
DEFAULT_V_MAX = 50.0  # km/h

# 신호등 정차 확률/시간
TL_STOP_PROB = 0.75
TL_BASE_SEC = 105.0
TL_JITTER_SEC = 75.0

# 정류장 기본 정차 시간(초)
STATION_DWELL_DEFAULT_SEC = 60.0

# 링크 불연속 보정 끔: “정류장/신호등 외에는 절대 서지 않음”
ENABLE_JOIN_CORRECTION = False
JOIN_GAP_THRESHOLD_M = 5.0  # (꺼져있으니 의미 없음)

# ⬇️ 정류장 조회 테이블/좌표계 명시
STATION_TABLE = "SIM_BIS_BUS_STATION_LOCATION"
STATION_XY_SRID = 4326  # x=lon, y=lat 이면 4326, (UTM-K 등) 투영좌표면 해당 SRID로 변경

# 정류장 스냅이 0/1에 달라붙는 것 완화용
EPS_RATIO = 1e-4

# 교통 기준일(링크별 vmax를 가져올 날짜). 시간대는 payload.departure_time의 '시' 사용
TRAFFIC_REF_DATE = _date(2025, 8, 1)

router = APIRouter()

# --------------------------------
# 공용: 안전한 departure_time 확보 (None 방지)
# --------------------------------
def _ensure_departure_time(dt: Optional[datetime]) -> datetime:
    """
    payload.departure_time 이 None 인 경우를 대비한 안전장치.
    규칙(임시): None -> (TRAFFIC_REF_DATE, 15:00)
    TODO(질문): None 일 때 어떤 기준을 쓸지 확정 필요.
    """
    if isinstance(dt, datetime):
        return dt.replace(microsecond=0)
    return datetime.combine(TRAFFIC_REF_DATE, _time(15, 0, 0))

# --------------------------------
# DB: 링크 geometry (WGS84) 가져오기
# --------------------------------
def _get_link_geometry_wgs84(db, link_id: int) -> LineString | None:
    q1 = text("""
        SELECT ST_AsText(ST_Transform(geometry, 4326)) AS wkt_geom
        FROM new_uroad
        WHERE "LINK_ID"::bigint = :lid
        LIMIT 1
    """)
    row = db.execute(q1, {"lid": int(link_id)}).fetchone()
    if row and row[0]:
        return _shp_wkt.loads(row[0])

    q2 = text("""
        SELECT ST_AsText(ST_Transform(geometry, 4326)) AS wkt_geom
        FROM new_uroad
        WHERE "LINK_ID"::text = :lid_text
        LIMIT 1
    """)
    row = db.execute(q2, {"lid_text": str(link_id)}).fetchone()
    if row and row[0]:
        return _shp_wkt.loads(row[0])

    return None

def _floor_to_5min_slot(dt: datetime) -> datetime:
    m = (dt.minute // 5) * 5
    return dt.replace(minute=m, second=0, microsecond=0)

# --------------------------------
# sim_traffic_congest 에서 roadname 기준 vmax(=MAX(avg_speed)) 조회
# - LINK_ID -> new_uroad."ROAD_NAME" 매핑
# - departure_time 을 5분 슬롯으로 내림하여 slot_5min 일치
# - 동일 roadname 복수 행이면 "최대 속도" 사용 (요구사항 반영)
# --------------------------------
def _fetch_per_link_vmax_from_congest_by_roadname(
    db: Session,
    link_ids: List[int],
    departure_time: Optional[datetime],
) -> Dict[int, float]:
    if not link_ids:
        return {}

    # 1) LINK_ID -> ROAD_NAME 매핑
    lids_text = [str(l) for l in link_ids]
    q_road = text("""
        SELECT "LINK_ID"::text AS lid_text, "ROAD_NAME"
        FROM new_uroad
        WHERE "LINK_ID"::text = ANY(:lid_list)
    """)
    rows = db.execute(q_road, {"lid_list": lids_text}).fetchall()

    lid_to_rn: Dict[int, str] = {}
    roadnames: List[str] = []
    for lid_text, rn in rows:
        try:
            lid_int = int(lid_text)
        except Exception:
            continue
        if rn is not None:
            _rn = str(rn).strip()
            if _rn != "":
                lid_to_rn[lid_int] = _rn
                roadnames.append(_rn)

    if not lid_to_rn:
        return {}

    # 유니크 로드네임만 전달 (ANY 성능/안정성)
    roadnames = list({r: None for r in roadnames}.keys())

    # 2) slot_5min 결정 (None 방지)
    base_dt = _ensure_departure_time(departure_time)
    slot_dt = _floor_to_5min_slot(base_dt)

    # 3) roadname & slot 매칭 → roadname 별 MAX(avg_speed) 사용
    #    NOTE: text() + ANY(:list) 는 psycopg2 가 배열로 적절히 바인딩함.
    q_congest = text("""
        SELECT roadname, MAX(avg_speed)::float AS vmax
        FROM sim_traffic_congest
        WHERE roadname = ANY(:rn_list)
          AND slot_5min = :slot_dt
        GROUP BY roadname
    """)
    crows = db.execute(q_congest, {"rn_list": roadnames, "slot_dt": slot_dt}).fetchall()

    rn_to_speed: Dict[str, float] = {}
    for rn, vmax in crows:
        if rn is None or vmax is None:
            continue
        try:
            v = float(vmax)
        except Exception:
            continue
        if v > 0.0:
            rn_to_speed[str(rn).strip()] = v

    # 4) 링크별 vmax 구성 (없으면 호출부에서 DEFAULT_V_MAX 사용)
    out: Dict[int, float] = {}
    for lid in link_ids:
        rn = lid_to_rn.get(int(lid))
        if not rn:
            continue
        v = rn_to_speed.get(rn)
        if v is not None and v > 0.0:
            out[int(lid)] = v
    return out

# --------------------------------
# station_list 기반 링크 시퀀스 & 정류장 정차 인덱스
#  - 세그먼트 경계의 단일 링크 중복 제거
# --------------------------------
def _build_links_and_station_stop_indices(
    db: Session, station_list: List[int], path_type: str
) -> tuple[List[int], Dict[int, int]]:
    full_links: List[int] = []
    stop_idx_to_station_order: Dict[int, int] = {}

    for i in range(len(station_list) - 1):
        seg_links = compute_links_via_router(
            db,
            start_station_id=station_list[i],
            end_station_id=station_list[i + 1],
            ptype=path_type,
        ) or []
        if not seg_links:
            continue

        if full_links and seg_links and full_links[-1] == seg_links[0]:
            seg_links = seg_links[1:]

        full_links.extend(seg_links)

        if full_links:
            stop_idx_to_station_order[len(full_links) - 1] = i + 1  # 1..N-1

    return full_links, stop_idx_to_station_order

# --------------------------------
# 굴곡도 계산 (절대 회전량 합 / 총거리[km])
# --------------------------------
def _polyline_turn_sum_radians(line: LineString) -> float:
    if line is None:
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

def _compute_route_curvature(db: Session, link_ids: List[int]) -> float:
    if not link_ids:
        return 0.0
    turn_sum = 0.0
    for lid in link_ids:
        line = _get_link_geometry_wgs84(db, lid)
        if line is not None:
            turn_sum += _polyline_turn_sum_radians(line)
    total_m = sum(float(compute_total_length([lid]) or 0.0) for lid in link_ids)
    total_km = max(1e-6, total_m / 1000.0)
    return round(turn_sum / total_km, 6)

# --------------------------------
# F_NODE/T_NODE 일괄 조회
# --------------------------------
def _fetch_link_nodes(db: Session, link_ids: List[int]) -> Dict[int, Tuple[int | None, int | None]]:
    if not link_ids:
        return {}
    lids_text = [str(l) for l in link_ids]
    q = text("""
        SELECT "LINK_ID"::text AS lid_text, "F_NODE", "T_NODE"
        FROM new_uroad
        WHERE "LINK_ID"::text = ANY(:lid_list)
    """)
    rows = db.execute(q, {"lid_list": lids_text}).fetchall()
    out: Dict[int, Tuple[int | None, int | None]] = {int(r[0]): (r[1], r[2]) for r in rows}
    for lid in link_ids:
        out.setdefault(int(lid), (None, None))
    return out

# --------------------------------
# (역주행 허용) 인접 교집합 + lookahead 기반 방향 결정
# --------------------------------
def _approx_dist_m(a_xy: tuple[float, float], b_xy: tuple[float, float]) -> float:
    ax, ay = a_xy
    bx, by = b_xy
    dx = (ax - bx) * np.cos(np.deg2rad((ay + by) * 0.5))
    dy = (ay - by)
    return float(np.hypot(dx, dy) * 111_320.0)

def _orient_links_by_intersection(db: Session, link_ids: List[int]) -> List[Tuple[int, bool]]:
    n = len(link_ids)
    if n == 0:
        return []

    nodes: Dict[int, Tuple[int | None, int | None]] = _fetch_link_nodes(db, link_ids)
    geoms: Dict[int, LineString | None] = {lid: _get_link_geometry_wgs84(db, lid) for lid in link_ids}

    def choose_flip_for_end(lid: int, end_target: int | None) -> bool:
        f, t = nodes.get(int(lid), (None, None))
        if end_target is None:
            return False
        if t == end_target:
            return False
        if f == end_target:
            return True
        return False

    def choose_flip_for_start(lid: int, start_target: int | None) -> bool:
        f, t = nodes.get(int(lid), (None, None))
        if start_target is None:
            return False
        if f == start_target:
            return False
        if t == start_target:
            return True
        return False

    flips: list[bool | None] = [None] * n

    i = 0
    while i < n:
        if i == n - 1:
            flips[i] = False if flips[i] is None else flips[i]
            break

        lid_i, lid_j = link_ids[i], link_ids[i + 1]
        fi, ti = nodes.get(int(lid_i), (None, None))
        fj, tj = nodes.get(int(lid_j), (None, None))

        common_ij = ({fi, ti}.intersection({fj, tj}) - {None})
        if common_ij:
            s = next(iter(common_ij))
            if flips[i] is None:
                flips[i] = choose_flip_for_end(lid_i, s)
            if flips[i + 1] is None:
                flips[i + 1] = choose_flip_for_start(lid_j, s)
            i += 1
            continue

        if i + 2 < n:
            lid_k = link_ids[i + 2]
            fk, tk = nodes.get(int(lid_k), (None, None))
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
    return list(zip(link_ids, flips))

# --------------------------------
# 속도/거리 유틸
# --------------------------------
def _ticks_from_seconds(sec: float, dt: float = DT) -> int:
    return max(0, int(round(sec / dt)))

def _cum_dists_from_speeds_kmh(speeds: List[float], dt: float = DT) -> np.ndarray:
    if not speeds:
        return np.array([], dtype=float)
    v = np.array(speeds, dtype=float) / 3.6  # m/s
    step = (v[:-1] + v[1:]) * 0.5 * dt
    cum = np.concatenate([[0.0], np.cumsum(step)])
    if len(cum) < len(speeds):
        cum = np.pad(cum, (0, len(speeds) - len(cum)), constant_values=cum[-1])
    return cum[: len(speeds)]

# --------------------------------
# ✨ 세그먼트 시뮬레이터 (감속/정지/연속성 보장)
# --------------------------------
def _simulate_segment_speeds(
    v_start_kmh: float,
    v_end_kmh: Optional[float],   # None이면 강제 종료속도 없음, 0이면 정지 목표
    v_max_kmh: float,
    distance_m: float,
    dt: float = DT,
) -> List[float]:
    v = max(0.0, v_start_kmh) / 3.6   # m/s
    vmax = max(0.0, v_max_kmh) / 3.6  # m/s
    vend = None if v_end_kmh is None else max(0.0, v_end_kmh) / 3.6  # m/s

    a_acc = 1.5  # m/s^2
    a_dec = 2.0  # m/s^2  → DT=0.1s일 때 tick당 0.2m/s 감소 = 0.72km/h

    speeds: List[float] = []
    dist = 0.0

    while dist < distance_m:
        # 제동 목표가 있을 때만 제동거리 고려
        if vend is not None and v > vend:
            braking_dist = (v**2 - vend**2) / (2 * a_dec)
        else:
            braking_dist = 0.0

        if vend is not None and v > vend and distance_m - dist <= braking_dist + 1e-9:
            a = -a_dec
            v_next = max(v + a * dt, vend)
        elif v < vmax:
            a = a_acc
            v_next = min(v + a * dt, vmax)
        else:
            a = 0.0
            v_next = v

        step = (v + v_next) * 0.5 * dt

        if dist + step > distance_m:
            if vend is None:
                speeds.append(round(v * 3.6, 2))
            else:
                rem = max(0.0, distance_m - dist)
                v_req = max(vend, (2.0 * rem / dt) - v)
                v_req = min(v_req, v)
                speeds.append(round(v_req * 3.6, 2))
            break

        dist += step
        v = v_next
        speeds.append(round(v * 3.6, 2))

        if vend is not None and abs(v - vend) < 1e-9 and dist >= distance_m - 1e-9:
            break

    if not speeds:
        last = v_end_kmh if v_end_kmh is not None else v_start_kmh
        speeds = [float(last)]

    return speeds

# --------------------------------
# 정류장 좌표 조회 (SIM_BIS_BUS_STATION_LOCATION → geometry 생성 → WGS84 좌표 반환)
# --------------------------------
def _fetch_station_points(db: Session, station_ids: List[int]) -> Dict[int, tuple[float, float]]:
    if not station_ids:
        return {}
    q = text(f"""
        SELECT
            station_id,
            ST_X(ST_Transform(ST_SetSRID(ST_MakePoint(x, y), :xy_srid), 4326)) AS lon,
            ST_Y(ST_Transform(ST_SetSRID(ST_MakePoint(x, y), :xy_srid), 4326)) AS lat
        FROM "{STATION_TABLE}"
        WHERE station_id = ANY(:sids)
    """)
    rows = db.execute(q, {"sids": station_ids, "xy_srid": STATION_XY_SRID}).fetchall()
    out: Dict[int, tuple[float, float]] = {}
    for sid, lon, lat in rows:
        if lon is not None and lat is not None:
            out[int(sid)] = (float(lon), float(lat))
    return out

# --------------------------------
# 정류장(중간) 정차를 링크 내부 위치(인덱스, ratio)로 매핑
# --------------------------------
def _map_stops_to_link_positions(
    db: Session,
    oriented_links: List[tuple[int, bool]],
    station_list: List[int],
) -> Dict[int, tuple[int, float]]:
    oriented_geoms: List[LineString | None] = []
    for lid, flip in oriented_links:
        g = _get_link_geometry_wgs84(db, lid)
        if g is not None and flip:
            g = LineString(list(g.coords)[::-1])
        oriented_geoms.append(g)

    station_ids_mid = station_list[1:]  # 출발 정류장 제외
    station_xy = _fetch_station_points(db, station_ids_mid)

    stop_pos: Dict[int, tuple[int, float]] = {}
    for order, sid in enumerate(station_list[1:], start=1):
        if sid not in station_xy:
            continue
        sx, sy = station_xy[sid]
        sp = _shp_wkt.loads(f"POINT({sx} {sy})")
        best_idx, best_d, best_ratio = -1, float("inf"), 0.0
        for idx, g in enumerate(oriented_geoms):
            if g is None or len(g.coords) < 2:
                continue
            d = g.distance(sp)
            if d < best_d:
                try:
                    r = g.project(sp, normalized=True)
                except Exception:
                    r = 0.0
                r = float(np.clip(r, 0.0 + EPS_RATIO, 1.0 - EPS_RATIO))
                best_idx, best_d, best_ratio = idx, d, r
        if best_idx >= 0:
            stop_pos[order] = (best_idx, best_ratio)

    return stop_pos

# --------------------------------
# (공용) 정차 블록 삽입: 설정 시간만큼 0 반복 + 좌표 고정
# --------------------------------
def _append_zero_block(speed_list: List[float], coord_list: List[tuple[float, float]], seconds: float):
    ticks = _ticks_from_seconds(seconds, dt=DT)
    if ticks <= 0:
        return
    last_coord = coord_list[-1] if coord_list else (0.0, 0.0)
    speed_list.extend([0.0] * ticks)
    coord_list.extend([last_coord] * ticks)

# --------------------------------
# NEW: uroad_traffic_filled에서 링크별 vmax(=actual_speed) 가져오기
#  - 기준일: TRAFFIC_REF_DATE(2025-08-01)
#  - 시간대: ref_hour(정수 시). createddate ∈ [ref_hour, ref_hour+1)
# --------------------------------
def _fetch_per_link_vmax_from_traffic(
    db: Session,
    link_ids: List[int],
    ref_hour: int,
) -> Dict[int, float]:
    if not link_ids:
        return {}
    start_dt = datetime.combine(TRAFFIC_REF_DATE, _time(ref_hour, 0, 0))
    end_dt = start_dt + timedelta(hours=1)

    lids_text = [str(l) for l in link_ids]
    q = text("""
        SELECT linkid::text AS lid_text, AVG(actual_speed)::float AS vmax_kmh
        FROM uroad_traffic_filled
        WHERE createddate >= :start_dt
          AND createddate <  :end_dt
          AND linkid::text = ANY(:lid_list)
        GROUP BY linkid::text
    """)
    rows = db.execute(q, {"start_dt": start_dt, "end_dt": end_dt, "lid_list": lids_text}).fetchall()

    out: Dict[int, float] = {}
    for lid_text, vmax_kmh in rows:
        try:
            lid_int = int(lid_text)
        except Exception:
            continue
        if vmax_kmh is not None and vmax_kmh > 0:
            out[lid_int] = float(vmax_kmh)
    return out

# --------------------------------
# 속도/좌표 생성 엔진 (링크별 vmax 적용)
#  - 비정지 구간 시작속도 0 리셋 방지
#  - 신호등은 링크 마지막 세그먼트를 ‘정지 목표’로 처리 후 대기
#  - 각 링크 vmax = 교통 테이블 값(없으면 DEFAULT_V_MAX)
#  - 🔧 보완: link_vmax 하한값 0.1km/h 적용(0 또는 음수 방지)
# --------------------------------
def _make_speed_and_coords(
    db: Session,
    link_ids: List[int],
    stop_idx_to_station_order: Dict[int, int],
    station_list: List[int],
    station_dwell_sec: List[float] | None = None,
    v_max_kmh: float = DEFAULT_V_MAX,
    p_stop_tl: float = TL_STOP_PROB,
    tl_base_sec: float = TL_BASE_SEC,
    tl_jitter_sec: float = TL_JITTER_SEC,
    seed: int | None = None,
    per_link_vmax: Optional[Dict[int, float]] = None,
) -> tuple[List[float], List[tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    speed_list: List[float] = []
    coord_list: List[tuple[float, float]] = []

    num_stops = max(stop_idx_to_station_order.values()) if stop_idx_to_station_order else 0
    if not station_dwell_sec:
        station_dwell_sec = [float(STATION_DWELL_DEFAULT_SEC)] * num_stops
    elif len(station_dwell_sec) < num_stops:
        station_dwell_sec = list(station_dwell_sec) + [station_dwell_sec[-1]] * (num_stops - len(station_dwell_sec))

    cur_v = 0.0
    len_cache: Dict[int, float] = {}

    oriented = _orient_links_by_intersection(db, link_ids)  # [(lid, flipped)]
    stop_positions = _map_stops_to_link_positions(db, oriented, station_list)

    geom_cache: Dict[int, LineString | None] = {}

    for idx, (lid, flip) in enumerate(oriented):
        if lid not in len_cache:
            len_cache[lid] = float(compute_total_length([lid]) or 0.0)
        link_len = len_cache[lid]
        if link_len <= 0:
            continue

        if lid not in geom_cache:
            base = _get_link_geometry_wgs84(db, lid)
            if base is not None and flip:
                base = LineString(list(base.coords)[::-1])
            geom_cache[lid] = base
        line = geom_cache[lid]

        # 이 링크의 vmax 결정 (교통값 우선, 없으면 기본값) + 하한 0.1km/h
        if per_link_vmax and lid in per_link_vmax:
            link_vmax = max(0.1, float(per_link_vmax[lid]))
        else:
            link_vmax = max(0.1, float(v_max_kmh))

        # 신호등 멈춤 여부 선결정
        try:
            tl_nodes = get_nodes_with_traffic_light(lid) or []
            has_tl = len(tl_nodes) > 0
        except Exception:
            has_tl = False
        tl_stop_now = has_tl and (p_stop_tl > 0.0) and (rng.random() < p_stop_tl)

        # 이 링크 내에서 “정류장 위치”로 분할
        inlink_orders: List[tuple[int, float]] = []
        for order, (lk_idx, ratio) in stop_positions.items():
            if lk_idx == idx:
                r = float(np.clip(ratio, 0.0 + EPS_RATIO, 1.0 - EPS_RATIO))
                inlink_orders.append((order, r))
        inlink_orders.sort(key=lambda x: x[1])

        cut_points = [0.0] + [r for _, r in inlink_orders] + [1.0]
        did_station_stop_on_link = False

        for seg_i in range(len(cut_points) - 1):
            r0, r1 = cut_points[seg_i], cut_points[seg_i + 1]
            if r1 <= r0:
                continue

            part_len = (r1 - r0) * link_len

            # 정류장 직전 세그먼트 or (신호등 정차 결정 시) 마지막 세그먼트를 정지 목표로
            is_last_segment = (seg_i == len(cut_points) - 2)
            is_before_station = seg_i < len(inlink_orders)
            is_before_stop = is_before_station or (tl_stop_now and is_last_segment)

            target_end_kmh: Optional[float] = 0.0 if is_before_stop else None

            # 구간 주행 시뮬레이션 (링크별 vmax 적용)
            seg_speeds = _simulate_segment_speeds(
                v_start_kmh=cur_v,
                v_end_kmh=target_end_kmh,
                v_max_kmh=link_vmax,
                distance_m=part_len,
                dt=DT,
            )
            speed_list.extend(seg_speeds)

            # 좌표 보간
            if line is not None and len(seg_speeds) > 0:
                if len(seg_speeds) == 1:
                    ratios = np.array([1.0], dtype=float)
                else:
                    ratios = np.linspace(0.0, 1.0, len(seg_speeds), endpoint=True)
                for rr in ratios:
                    R = r0 + (r1 - r0) * float(rr)
                    pt = line.interpolate(R, normalized=True)
                    coord_list.append((float(pt.x), float(pt.y)))
            else:
                coord_list.extend([(0.0, 0.0) for _ in range(len(seg_speeds))])

            if seg_speeds:
                cur_v = float(seg_speeds[-1])

            if is_before_station:
                order = inlink_orders[seg_i][0]
                dwell_sec = float(
                    station_dwell_sec[order - 1]
                    if 1 <= order <= len(station_dwell_sec)
                    else STATION_DWELL_DEFAULT_SEC
                )
                _append_zero_block(speed_list, coord_list, dwell_sec)
                cur_v = 0.0
                did_station_stop_on_link = True

        # 신호등 대기(마지막 세그먼트에서 이미 0까지 감속)
        if tl_stop_now and not did_station_stop_on_link:
            dwell = tl_base_sec + rng.uniform(-tl_jitter_sec, tl_jitter_sec)
            _append_zero_block(speed_list, coord_list, max(0.0, dwell))
            cur_v = 0.0

    return speed_list, coord_list

def post_speed_list_adjust_backward(speed_list: List[float], a_dec: float = 0.72) -> List[float]:
    n = len(speed_list)
    if n <= 1:
        return speed_list
    s = list(speed_list)
    for i in range(n - 2, -1, -1):
        cap = s[i + 1] + a_dec
        if s[i] > cap:
            s[i] = round(cap, 2)
    return s

def rebuild_coords_from_speed_list(
    db,
    link_ids: List[int],
    speed_list: List[float],
    dt: float = DT,  # 0.1s
) -> List[Tuple[float, float]]:
    # 1) 링크를 실제 진행 방향으로 정렬/뒤집기
    oriented = _orient_links_by_intersection(db, link_ids)  # [(lid, flip)]

    # 2) 링크별 geometry와 실제 길이(m) 확보
    geoms: List[LineString] = []
    lens_m: List[float] = []
    for lid, flip in oriented:
        g = _get_link_geometry_wgs84(db, lid)
        if g is None or len(g.coords) < 2:
            continue
        if flip:
            g = LineString(list(g.coords)[::-1])
        L = float(compute_total_length([lid]) or 0.0)
        if L <= 0.0:
            continue
        geoms.append(g)
        lens_m.append(L)
    if not geoms:
        return [(0.0, 0.0)] * len(speed_list)

    cum_link_m = np.cumsum(lens_m)  # 각 링크 끝까지의 누적(m)
    total_path_m = float(cum_link_m[-1])

    # 3) 바뀐 speed_list로부터 틱별 누적 이동거리(m) 계산
    def _cum_dists_from_speeds_kmh_local(speeds: List[float], dt: float) -> np.ndarray:
        if not speeds:
            return np.array([], dtype=float)
        v = np.array(speeds, dtype=float) / 3.6  # m/s
        if len(v) == 1:
            return np.array([0.0], dtype=float)
        step = (v[:-1] + v[1:]) * 0.5 * dt
        cum = np.concatenate([[0.0], np.cumsum(step)])
        return cum

    cum_m = _cum_dists_from_speeds_kmh_local(speed_list, dt=dt)
    # 경로 총길이 초과분은 말단에 클램프
    cum_m = np.clip(cum_m, 0.0, total_path_m)

    # 4) 각 누적거리 위치를 (어느 링크, 링크 내 ratio)로 변환 후 좌표 샘플
    coords: List[Tuple[float, float]] = []
    li = 0  # 현재 링크 인덱스 포인터
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

# --------------------------------
# FastAPI 엔드포인트
# --------------------------------
@router.post("/", response_model=ScenarioOut, status_code=status.HTTP_201_CREATED)
def create_scenario(payload: ScenarioCreate, db: Session = Depends(get_db)):
    route = db.execute(
        select(Route).where(Route.route_id == payload.route_id)
    ).scalar_one_or_none()
    if not route:
        raise HTTPException(status_code=404, detail="해당 route_id가 존재하지 않습니다.")

    if payload.path_type not in ("shortest", "optimal"):
        raise HTTPException(status_code=400, detail="path_type은 shortest/optimal 중 하나여야 합니다.")

    station_list: List[int] = route.station_list or []
    if len(station_list) < 2:
        raise HTTPException(status_code=400, detail="해당 노선의 station_list가 비어있습니다.")

    link_list, stop_idx_to_station_order = _build_links_and_station_stop_indices(
        db, station_list, payload.path_type
    )
    if not link_list:
        raise HTTPException(status_code=400, detail="유효한 링크 경로를 생성하지 못했습니다.")

    route_length_m = float(sum(float(compute_total_length([lid]) or 0.0) for lid in link_list))
    route_curvature = _compute_route_curvature(db, link_list)

    # 안전한 departure_time 확보 (None 방지)
    safe_departure = _ensure_departure_time(payload.departure_time)

    # 링크별 vmax(=최대 avg_speed) 조회: sim_traffic_congest 기준 (roadname 매칭)
    per_link_vmax = _fetch_per_link_vmax_from_congest_by_roadname(
        db=db,
        link_ids=link_list,
        departure_time=safe_departure,
    )

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
        per_link_vmax=per_link_vmax,
    )

    # 후방 감속 캡 보정 + 좌표 재구성(물리 일관성)
    speed_list = post_speed_list_adjust_backward(speed_list, a_dec=0.72)
    coord_list = rebuild_coords_from_speed_list(
        db=db,
        link_ids=link_list,
        speed_list=speed_list,
        dt=DT,
    )

    scenario = Scenario(
        name=payload.name,
        route_id=payload.route_id,
        headway_min=payload.headway_min,
        start_time=payload.start_time,
        end_time=payload.end_time,
        departure_time=safe_departure,
        path_type=payload.path_type,
        route_length=route_length_m,
        route_curvature=route_curvature,
        speed_list=speed_list,
        coord_list=coord_list,
        link_list=link_list,
    )
    db.add(scenario)
    db.commit()
    db.refresh(scenario)

    return ScenarioOut(
        scenario_id=scenario.scenario_id,
        name=scenario.name,
        route_id=scenario.route_id,
        headway_min=scenario.headway_min,
        start_time=scenario.start_time,
        end_time=scenario.end_time,
        departure_time=scenario.departure_time,
        path_type=scenario.path_type,
        route_length=scenario.route_length,
        route_curvature=scenario.route_curvature,
        speed_list=scenario.speed_list,
        coord_list=scenario.coord_list,
        link_list=scenario.link_list,
    )
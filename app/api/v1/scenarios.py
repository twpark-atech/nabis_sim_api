# app/api/v1/scenarios.py
# ✅ 요청대로: 코드만 제공합니다. (정지는 ‘정류장’과 ‘신호등 있는 노드’에서만 발생)
from __future__ import annotations

from typing import List, Tuple, Dict, Optional
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
TL_STOP_PROB = 0.0
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

router = APIRouter()

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
# ✨ 수정된 세그먼트 시뮬레이터
#  - v_end_kmh가 None이면 끝속도 강제 고정하지 않음(평상 주행 구간)
#  - 정차 직전 구간만 v_end_kmh=0.0으로 강제
#  - 링크 경계에서 ‘잔여거리 맞춤용 급감속 샘플’ 대신, 연속 속도 유지(다음 링크에 이월)
# --------------------------------
def _simulate_segment_speeds(
    v_start_kmh: float,
    v_end_kmh: Optional[float],   # ← None이면 끝속도 강제 고정 안 함
    v_max_kmh: float,
    distance_m: float,
    dt: float = DT,
) -> List[float]:
    v = max(0.0, v_start_kmh) / 3.6
    vmax = max(0.0, v_max_kmh) / 3.6
    vend = None if v_end_kmh is None else max(0.0, v_end_kmh) / 3.6

    a_acc = 1.5  # m/s^2
    a_dec = 2.0  # m/s^2

    speeds: List[float] = []
    dist = 0.0
    while dist < distance_m:
        # 제동 목표가 있을 때만 제동거리 고려
        if vend is not None and v > vend:
            braking_dist = (v**2 - vend**2) / (2 * a_dec)
        else:
            braking_dist = 0.0

        if vend is not None and distance_m - dist <= braking_dist:
            # 정지 목표가 있을 때만 감속에 들어감
            a = -a_dec
            v_next = max(0.0, min(v + a * dt, vmax))
            step = (v + v_next) * 0.5 * dt
            if dist + step > distance_m:
                # 남은 거리만큼 정확히 채우는 보정 속도 (정지 케이스에만 사용)
                rem = distance_m - dist
                v_star = max(0.0, (2 * rem / dt) - v)
                speeds.append(v_star * 3.6)
                break
            dist += step
            v = v_next
            speeds.append(v * 3.6)
            continue

        # 일반 구간(끝속도 강제 X): 가속/정속
        a = a_acc if v < vmax else 0.0
        v_next = max(0.0, min(v + a * dt, vmax))
        step = (v + v_next) * 0.5 * dt

        if dist + step > distance_m:
            # 🚫 더 이상 샘플을 억지로 끼워 넣지 않음
            #    → 링크 경계에서 급감속 샘플(작은 속도) 생성 방지
            #    → 다음 링크 시작속도로 현재 속도 v 이월
            speeds.append(v * 3.6)
            break

        dist += step
        v = v_next
        speeds.append(v * 3.6)

    # 끝속도를 강제해야 할 때만 마지막 샘플을 고정(정차 직전 세그먼트)
    if speeds and v_end_kmh is not None:
        speeds[-1] = float(v_end_kmh)
    elif not speeds:
        # 거리 매우 짧은 경우 보호
        speeds = [float(v_end_kmh if v_end_kmh is not None else v_start_kmh)]

    return [round(s, 2) for s in speeds]


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
                # 끝점 달라붙음 완화
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
# 속도/좌표 생성 엔진
#  - “정류장/신호등”에서만 정지
#  - 링크 불연속 보정/단발 0/속도리셋 없음
#  - ✅ 링크 경계에서 속도 연속성 유지(다음 LINK 시작속도에 반영)
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

        # 🔴 링크 불연속 보정 완전 비활성화: 정류장/신호등 외에는 절대 0을 넣지 않음

        # 이 링크 내에서 “정류장 위치”로 분할
        inlink_orders: List[tuple[int, float]] = []
        for order, (lk_idx, ratio) in stop_positions.items():
            if lk_idx == idx:
                r = float(np.clip(ratio, 0.0 + EPS_RATIO, 1.0 - EPS_RATIO))
                inlink_orders.append((order, r))
        inlink_orders.sort(key=lambda x: x[1])

        # 이 링크가 신호등 노드를 갖는지
        try:
            tl_nodes = get_nodes_with_traffic_light(lid) or []  # 함수 내부에서 str 캐스팅 처리
            has_tl = len(tl_nodes) > 0
        except Exception:
            has_tl = False

        cut_points = [0.0] + [r for _, r in inlink_orders] + [1.0]

        for seg_i in range(len(cut_points) - 1):
            r0, r1 = cut_points[seg_i], cut_points[seg_i + 1]
            if r1 <= r0:
                continue

            part_len = (r1 - r0) * link_len

            # ✨ 정류장 직전 세그먼트만 '정지' 목표, 그 외는 끝속도 강제 없음
            is_before_stop = seg_i < len(inlink_orders)
            target_end_kmh: Optional[float] = 0.0 if is_before_stop else None

            # 구간 주행 시뮬레이션
            seg_speeds = _simulate_segment_speeds(
                v_start_kmh=cur_v,
                v_end_kmh=target_end_kmh,
                v_max_kmh=v_max_kmh,
                distance_m=part_len,
                dt=DT,
            )
            speed_list.extend(seg_speeds)

            # 좌표 보간: 샘플 수에 맞춰 균등 분할(항상 r1까지 도달)
            if line is not None and len(seg_speeds) > 0:
                ratios = np.linspace(0.0, 1.0, len(seg_speeds), endpoint=True)
                for rr in ratios:
                    R = r0 + (r1 - r0) * float(rr)
                    pt = line.interpolate(R, normalized=True)
                    coord_list.append((float(pt.x), float(pt.y)))
            else:
                coord_list.extend([(0.0, 0.0) for _ in range(len(seg_speeds))])

            # ✨ 다음 세그먼트 시작속도는 “실제 마지막 속도”로 이월
            if seg_speeds:
                cur_v = float(seg_speeds[-1])

            # 정류장 정차: 설정한 시간만큼 0 반복 & 좌표 고정
            if is_before_stop:
                order = inlink_orders[seg_i][0]
                dwell_sec = float(
                    station_dwell_sec[order - 1]
                    if 1 <= order <= len(station_dwell_sec)
                    else STATION_DWELL_DEFAULT_SEC
                )
                _append_zero_block(speed_list, coord_list, dwell_sec)
                cur_v = 0.0  # 정차 후 재출발

        # 신호등 정차: 확률(p_stop_tl)이 0.0이면 절대 서지 않음
        if has_tl and (p_stop_tl > 0.0) and (rng.random() < p_stop_tl):
            dwell = tl_base_sec + rng.uniform(-tl_jitter_sec, tl_jitter_sec)
            _append_zero_block(speed_list, coord_list, max(0.0, dwell))
            cur_v = 0.0

    return speed_list, coord_list


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

    # ⬇️ 정류장 1분 정차, 신호등 확률은 요청값 사용(0.0이면 TL 정차 절대 없음)
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

    scenario = Scenario(
        name=payload.name,
        route_id=payload.route_id,
        headway_min=payload.headway_min,
        start_time=payload.start_time,
        end_time=payload.end_time,
        departure_time=payload.departure_time,
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

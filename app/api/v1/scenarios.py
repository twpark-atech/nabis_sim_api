# app/api/v1/scenarios.py

from __future__ import annotations

from typing import List, Tuple, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, text
from sqlalchemy.orm import Session
from shapely.geometry import LineString
from shapely import wkt as _shp_wkt
import numpy as np
import pandas as pd

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
TL_STOP_PROB = 0.75
TL_BASE_SEC = 105.0
TL_JITTER_SEC = 75.0
DEFAULT_V_MAX = 50.0  # km/h

router = APIRouter()

# --------------------------------
# DB: 링크 geometry (WGS84) 가져오기
# --------------------------------
def _get_link_geometry_wgs84(db, link_id: int) -> LineString | None:
    """
    new_uroad.geometry(LineString)를 WGS84로 가져와 shapely LineString 반환.
    - new_uroad."LINK_ID" 가 TEXT 인 환경에서도 동작하도록 캐스팅 처리.
    """
    # 1차 시도: 컬럼을 ::bigint 로 캐스팅 (TEXT 컬럼이어도 숫자형 텍스트라면 OK)
    q1 = text("""
        SELECT ST_AsText(ST_Transform(geometry, 4326)) AS wkt_geom
        FROM new_uroad
        WHERE "LINK_ID"::bigint = :lid
        LIMIT 1
    """)
    row = db.execute(q1, {"lid": int(link_id)}).fetchone()
    if row and row[0]:
        return _shp_wkt.loads(row[0])

    # 2차 시도: 파라미터를 TEXT 로 비교 (컬럼이 TEXT 고, 숫자 이외의 값 가능성 대비)
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

        # ✅ 경계 중복 제거: 앞 세그먼트의 마지막 링크 == 이번 세그먼트의 첫 링크면 첫 링크 제거
        if full_links and seg_links and full_links[-1] == seg_links[0]:
            seg_links = seg_links[1:]

        full_links.extend(seg_links)

        # 정류장 정차 인덱스: 붙인 뒤의 마지막 링크 인덱스에 찍기
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
# NEW ①: F_NODE/T_NODE 일괄 조회
# --------------------------------
def _fetch_link_nodes(db: Session, link_ids: List[int]) -> Dict[int, Tuple[int | None, int | None]]:
    """
    반환: {lid: (F_NODE, T_NODE)} — 누락 시 (None, None)
    """
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
# NEW ②: (역주행 허용) 인접 교집합 + lookahead 기반 방향 결정
#  - (i, i+1)가 불연속이라도 (i+1, i+2)가 연결되면 i+1을 그 접속점으로 끝내고,
#    i는 i+1의 시작점으로 끝나도록 보정
# --------------------------------
def _approx_dist_m(a_xy: tuple[float, float], b_xy: tuple[float, float]) -> float:
    ax, ay = a_xy
    bx, by = b_xy
    dx = (ax - bx) * np.cos(np.deg2rad((ay + by) * 0.5))
    dy = (ay - by)
    return float(np.hypot(dx, dy) * 111_320.0)


def _orient_links_by_intersection(db: Session, link_ids: List[int]) -> List[Tuple[int, bool]]:
    """
    [(lid, flipped)] 반환.
    기본: (i, i+1) 교집합 노드 s를 기준으로 i는 s에서 끝, i+1은 s에서 시작.
    특수: (i, i+1) 불연속이고 (i+1, i+2)에 교집합이 있으면
          · i+1은 그 교집합(s12)에서 '끝나도록'(역주행 허용)
          · i는 i+1의 '시작'으로 끝나도록 보정
    """
    n = len(link_ids)
    if n == 0:
        return []

    nodes: Dict[int, Tuple[int | None, int | None]] = _fetch_link_nodes(db, link_ids)
    geoms: Dict[int, LineString | None] = {lid: _get_link_geometry_wgs84(db, lid) for lid in link_ids}

    def endpoints(lid: int, flip: bool) -> tuple[int | None, int | None]:
        f, t = nodes.get(int(lid), (None, None))
        return (t, f) if flip else (f, t)  # (start, end)

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

        # (i, i+1) 교집합
        common_ij = ({fi, ti}.intersection({fj, tj}) - {None})
        if common_ij:
            s = next(iter(common_ij))
            if flips[i] is None:
                flips[i] = choose_flip_for_end(lid_i, s)
            if flips[i + 1] is None:
                flips[i + 1] = choose_flip_for_start(lid_j, s)
            i += 1
            continue

        # lookahead: (i+1, i+2) 교집합으로 i+1 끝점을 맞추고 i를 그 시작점에 맞춤
        if i + 2 < n:
            lid_k = link_ids[i + 2]
            fk, tk = nodes.get(int(lid_k), (None, None))
            common_jk = ({fj, tj}.intersection({fk, tk}) - {None})
            if common_jk:
                s12 = next(iter(common_jk))
                flips[i + 1] = choose_flip_for_end(lid_j, s12)

                # i는 i+1의 시작으로 끝나게
                gj = geoms.get(lid_j)
                if flips[i + 1]:
                    start_j = tj  # flipped: start is original T
                else:
                    start_j = fj

                if flips[i] is None:
                    if start_j in {fi, ti}:
                        flips[i] = choose_flip_for_end(lid_i, start_j)
                    else:
                        gi = geoms.get(lid_i)
                        if gi is not None and gj is not None and len(gi.coords) >= 2 and len(gj.coords) >= 2:
                            start_j_xy = (gj.coords[0][0], gj.coords[0][1]) if not flips[i + 1] else (gj.coords[-1][0], gj.coords[-1][1])
                            end_i_xy_false = (gi.coords[-1][0], gi.coords[-1][1])
                            end_i_xy_true = (gi.coords[0][0], gi.coords[0][1])
                            flips[i] = (_approx_dist_m(end_i_xy_true, start_j_xy) <
                                        _approx_dist_m(end_i_xy_false, start_j_xy))
                        else:
                            flips[i] = False
                i += 1
                continue

        # 여전히 정보 부족 → 지오메트리 근접도로 i만 결정
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


def _simulate_segment_speeds(
    v_start_kmh: float, v_end_kmh: float, v_max_kmh: float, distance_m: float, dt: float = DT
) -> List[float]:
    v = v_start_kmh / 3.6
    vmax = v_max_kmh / 3.6
    vend = v_end_kmh / 3.6
    a_acc = 1.5  # m/s^2
    a_dec = 2.0  # m/s^2

    speeds: List[float] = []
    dist = 0.0
    while dist < distance_m:
        braking_dist = (v**2 - vend**2) / (2 * a_dec) if v > vend else 0.0
        if distance_m - dist <= braking_dist:
            a = -a_dec
        elif v < vmax:
            a = a_acc
        else:
            a = 0.0

        v_next = max(0.0, min(v + a * dt, vmax))
        step = (v + v_next) * 0.5 * dt
        if dist + step > distance_m:
            rem = distance_m - dist
            v_star = max(0.0, (2 * rem / dt) - v)
            speeds.append(v_star * 3.6)
            break

        dist += step
        v = v_next
        speeds.append(v * 3.6)

    if speeds:
        speeds[-1] = float(v_end_kmh)
    else:
        speeds = [float(v_end_kmh)]
    return [round(s, 2) for s in speeds]


# --------------------------------
# 속도/좌표 생성 엔진
#  - 링크 방향 보정(역주행 허용) + 불연속 direct 연결
# --------------------------------
def _make_speed_and_coords(
    db: Session,
    link_ids: List[int],
    stop_idx_to_station_order: Dict[int, int],
    station_dwell_sec: List[float] | None = None,
    v_max_kmh: float = DEFAULT_V_MAX,
    p_stop_tl: float = TL_STOP_PROB,
    tl_base_sec: float = TL_BASE_SEC,
    tl_jitter_sec: float = TL_JITTER_SEC,
    seed: int | None = None,
) -> tuple[List[float], List[tuple[float, float]]]:
    """
    링크별 주행을 순회하며 속도/좌표 시퀀스를 만든다.
    - 정류장: stop_idx_to_station_order 에 명시된 링크 끝에서 정차
    - 신호등: 확률적으로 정차(105±75초)
    - 방향: 인접 교집합 + lookahead 기반으로 보정(역주행 허용)
    - 불연속 접속: 이전 끝점과 다음 시작점이 멀면 'direct'로 1틱(0 m/s) 이어붙임
    """
    rng = np.random.default_rng(seed)
    speed_list: List[float] = []
    coord_list: List[tuple[float, float]] = []

    # 정류장 정차시간 준비 (출발 제외)
    num_stops = max(stop_idx_to_station_order.values()) if stop_idx_to_station_order else 0
    if not station_dwell_sec:
        station_dwell_sec = [20.0] * num_stops
    elif len(station_dwell_sec) < num_stops:
        station_dwell_sec = list(station_dwell_sec) + [station_dwell_sec[-1]] * (num_stops - len(station_dwell_sec))

    cur_v = 0.0
    len_cache: Dict[int, float] = {}

    # 방향 정합 계산
    oriented = _orient_links_by_intersection(db, link_ids)  # [(lid, flipped)]
    geom_cache: Dict[int, LineString | None] = {}

    for idx, (lid, flip) in enumerate(oriented):
        if lid not in len_cache:
            len_cache[lid] = float(compute_total_length([lid]) or 0.0)
        link_len = len_cache[lid]
        if link_len <= 0:
            continue

        # geometry (필요 시 뒤집기)
        if lid not in geom_cache:
            base = _get_link_geometry_wgs84(db, lid)
            if base is not None and flip:
                base = LineString(list(base.coords)[::-1])
            geom_cache[lid] = base
        line = geom_cache[lid]

        # ✅ 불연속 direct 연결: 이전 끝점과 이번 시작점이 멀면 1틱(속도 0)으로 잇기
        if coord_list and line is not None and len(line.coords) >= 2:
            start_xy = (line.coords[0][0], line.coords[0][1])
            if _approx_dist_m(coord_list[-1], start_xy) > 5.0:  # 5m 초과면 direct 패치
                speed_list.append(0.0)
                coord_list.append(start_xy)
                cur_v = 0.0

        # 정류장 / 신호등 정차 판단
        station_stop_sec = 0.0
        if idx in stop_idx_to_station_order:
            stop_order = stop_idx_to_station_order[idx]  # 1..N-1
            station_stop_sec = float(station_dwell_sec[stop_order - 1])

        try:
            tl_nodes = get_nodes_with_traffic_light(lid)
            has_tl = bool(tl_nodes)
        except Exception:
            has_tl = False

        tl_stop_sec = 0.0
        if has_tl and (rng.random() < p_stop_tl):
            dwell = tl_base_sec + rng.uniform(-tl_jitter_sec, tl_jitter_sec)
            tl_stop_sec = max(0.0, dwell)

        will_stop = (station_stop_sec > 0.0) or (tl_stop_sec > 0.0)
        v_end = 0.0 if will_stop else min(v_max_kmh, max(0.0, cur_v))

        # 주행 구간 속도
        seg_speeds = _simulate_segment_speeds(cur_v, v_end, v_max_kmh, link_len, dt=DT)
        speed_list.extend(seg_speeds)

        # 좌표 보간
        cum_m = _cum_dists_from_speeds_kmh(seg_speeds, dt=DT)
        ratios = (cum_m / max(1e-6, link_len)).clip(0.0, 1.0)
        if line is not None:
            for r in ratios:
                pt = line.interpolate(float(r), normalized=True)
                coord_list.append((float(pt.x), float(pt.y)))
        else:
            # geometry가 없으면 (0,0)으로 채움
            coord_list.extend([(0.0, 0.0) for _ in range(len(seg_speeds))])

        cur_v = v_end

        # 정차 구간(속도=0, 좌표 고정)
        dwell_total_sec = station_stop_sec + tl_stop_sec
        ticks = _ticks_from_seconds(dwell_total_sec, dt=DT)
        if ticks > 0:
            speed_list.extend([0.0] * ticks)
            last_coord = coord_list[-1] if coord_list else (0.0, 0.0)
            coord_list.extend([last_coord] * ticks)
            cur_v = 0.0  # 정차 후 재출발

    return speed_list, coord_list


# --------------------------------
# FastAPI 엔드포인트
# --------------------------------
@router.post("/", response_model=ScenarioOut, status_code=status.HTTP_201_CREATED)
def create_scenario(payload: ScenarioCreate, db: Session = Depends(get_db)):
    """
    Input:  name(시나리오명), route_id, headway_min(배차시간), start_time, end_time,
            departure_time, path_type
    Output(DB): scenario_id, name, route_id, headway_min, start_time, end_time,
                departure_time, path_type, route_length, route_curvature,
                speed_list, coord_list
    """
    # 1) 노선 확인
    route = db.execute(
        select(Route).where(Route.route_id == payload.route_id)
    ).scalar_one_or_none()
    if not route:
        raise HTTPException(status_code=404, detail="해당 route_id가 존재하지 않습니다.")

    # 2) path_type 검증
    if payload.path_type not in ("shortest", "optimal"):
        raise HTTPException(status_code=400, detail="path_type은 shortest/optimal 중 하나여야 합니다.")

    # 3) station_list 확보
    station_list: List[int] = route.station_list or []
    if len(station_list) < 2:
        raise HTTPException(status_code=400, detail="해당 노선의 station_list가 비어있습니다.")

    # 4) 전체 링크 & 정류장 정차 인덱스 (경계 중복 제거 포함)
    link_list, stop_idx_to_station_order = _build_links_and_station_stop_indices(
        db, station_list, payload.path_type
    )
    if not link_list:
        raise HTTPException(status_code=400, detail="유효한 링크 경로를 생성하지 못했습니다.")

    # 5) 길이/굴곡도
    route_length_m = float(sum(float(compute_total_length([lid]) or 0.0) for lid in link_list))
    route_curvature = _compute_route_curvature(db, link_list)

    # 6) 속도/좌표 시퀀스 생성 (방향 보정/불연속 direct 포함)
    speed_list, coord_list = _make_speed_and_coords(
        db=db,
        link_ids=link_list,
        stop_idx_to_station_order=stop_idx_to_station_order,
        station_dwell_sec=None,             # 필요 시 payload에 추가해 전달
        v_max_kmh=DEFAULT_V_MAX,            # 필요 시 payload로 공개
        p_stop_tl=TL_STOP_PROB,
        tl_base_sec=TL_BASE_SEC,
        tl_jitter_sec=TL_JITTER_SEC,
        seed=42,
    )

    # 7) 저장
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
        speed_list=speed_list,         # JSON 컬럼 권장
        coord_list=coord_list,         # JSON 컬럼 권장: [(lon,lat), ...]
        link_list=link_list,           # 필요 시 저장
    )
    db.add(scenario)
    db.commit()
    db.refresh(scenario)

    # 8) 응답
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

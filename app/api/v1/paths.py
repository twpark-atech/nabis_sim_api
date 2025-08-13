# app/api/v1/paths.py  # 기존 파일 교체

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Response, status
from shapely import wkt as _wkt
from shapely.geometry import LineString, Point
from sqlalchemy import and_, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from pyproj import Transformer

from app.db import engine, get_db
from app.models import Path, Station
from app.schemas import PathCreate, PathOut, PathType

router = APIRouter()

# ------------------------------------------------------------------------------------
# 환경 변수 / 상수
# ------------------------------------------------------------------------------------
UROAD_TABLE = os.getenv("UROAD_SOURCE_TABLE", "new_uroad")  # 링크 원천 테이블(라인스트링)
SRID_WGS84 = int(os.getenv("SRID_WGS84", "4326"))
SRID_METRIC = int(os.getenv("SRID_METRIC", "5179"))  # 미터 좌표계(EPSG:5179 권장)
SANITY_MAX_M = float(os.getenv("SNAP_SANITY_MAX_M", "2000"))  # 좌표-도로 최근접 허용 최대치(m)
BBOX_BUFFER_M = float(os.getenv("BBOX_BUFFER_M", "12000"))    # 출발/도착 주변 BBOX 버퍼(m)

# ID 범위 정책(기존 요구사항 유지)
EXIST_MIN, EXIST_MAX = 600_000_000, 630_000_000  # [min, max)
SHORTEST_MIN, SHORTEST_MAX = 630_000_000, 660_000_000
OPTIMAL_MIN, OPTIMAL_MAX = 660_000_000, 690_000_000


# ------------------------------------------------------------------------------------
# ID 발급/조회
# ------------------------------------------------------------------------------------
def _range_for(ptype: PathType) -> Tuple[int, int]:
    if ptype == "existing":
        return EXIST_MIN, EXIST_MAX
    if ptype == "shortest":
        return SHORTEST_MIN, SHORTEST_MAX
    return OPTIMAL_MIN, OPTIMAL_MAX


def _next_path_id(db: Session, ptype: PathType) -> int:
    lo, hi = _range_for(ptype)
    max_id = db.execute(
        select(func.max(Path.path_id)).where(and_(Path.path_id >= lo, Path.path_id < hi))
    ).scalar()
    nxt = lo if max_id is None else int(max_id) + 1
    if nxt >= hi:
        raise HTTPException(status_code=409, detail=f"{ptype} path_id capacity exhausted")
    return nxt


def _find_path(db: Session, start_id: int, end_id: int, ptype: PathType) -> Optional[Path]:
    lo, hi = _range_for(ptype)
    return (
        db.execute(
            select(Path)
            .where(
                and_(
                    Path.start_station_id == start_id,
                    Path.end_station_id == end_id,
                    Path.path_id >= lo,
                    Path.path_id < hi,
                )
            )
            .limit(1)
        )
        .scalars()
        .first()
    )


# ------------------------------------------------------------------------------------
# 좌표 변환
# ------------------------------------------------------------------------------------
def _to_xy(lon: float, lat: float, transformer: Transformer) -> Tuple[float, float]:
    x, y = transformer.transform(lon, lat)  # always_xy=True
    return float(x), float(y)


def make_bbox_metric_around(
    origin_lon: float,
    origin_lat: float,
    dest_lon: float,
    dest_lat: float,
    buffer_m: float = BBOX_BUFFER_M,
) -> Tuple[float, float, float, float]:
    to_metric = Transformer.from_crs(SRID_WGS84, SRID_METRIC, always_xy=True)
    ox, oy = _to_xy(origin_lon, origin_lat, to_metric)
    dx, dy = _to_xy(dest_lon, dest_lat, to_metric)
    xmin = min(ox, dx) - buffer_m
    xmax = max(ox, dx) + buffer_m
    ymin = min(oy, dy) - buffer_m
    ymax = max(oy, dy) + buffer_m
    return xmin, xmax, ymin, ymax


# ------------------------------------------------------------------------------------
# 링크 로드(new_uroad 등)
# ------------------------------------------------------------------------------------
def _get_single_srid(engine, tbl: str) -> int:
    with engine.begin() as conn:
        df = pd.read_sql(f"SELECT DISTINCT ST_SRID(geometry) AS srid FROM {tbl}", conn)
    if df.empty:
        raise ValueError(f"{tbl}: geometry/SRID 확인 실패")
    srids = df["srid"].dropna().astype(int).unique().tolist()
    if len(srids) != 1:
        raise ValueError(f"{tbl}: 혼합 SRID 감지 {srids}")
    if srids[0] == 0:
        raise ValueError(f"{tbl}: geometry SRID=0 (미지정)")
    return int(srids[0])


def load_links_from_postgis(
    engine,
    table: str = UROAD_TABLE,
    schema: Optional[str] = None,
    srid_metric: int = SRID_METRIC,
    bbox_metric: Optional[Tuple[float, float, float, float]] = None,  # (xmin,xmax,ymin,ymax)
) -> pd.DataFrame:
    tbl = table if schema is None else f"{schema}.{table}"
    src_srid = _get_single_srid(engine, tbl)
    geom_expr = "geometry" if src_srid == srid_metric else f"ST_Transform(geometry, {srid_metric})"

    where = ""
    if bbox_metric is not None:
        xmin, xmax, ymin, ymax = bbox_metric
        where = f"""
        WHERE ST_Intersects(
          {geom_expr},
          ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax}, {srid_metric})
        )
        """

    sql = f"""
    SELECT
      "LINK_ID","F_NODE","T_NODE",
      COALESCE("MAX_SPD", NULL) AS max_spd,
      COALESCE("ROAD_RANK", NULL) AS road_rank,
      COALESCE("ROAD_TYPE", NULL) AS road_type,
      COALESCE("LANES", NULL)    AS lanes,
      ST_AsText({geom_expr}) AS wkt_5179,
      ST_Length({geom_expr}) AS length_m,
      ST_X(ST_StartPoint({geom_expr})) AS start_x,
      ST_Y(ST_StartPoint({geom_expr})) AS start_y,
      ST_X(ST_EndPoint({geom_expr}))   AS end_x,
      ST_Y(ST_EndPoint({geom_expr}))   AS end_y
    FROM {tbl}
    {where};
    """
    with engine.begin() as conn:
        df = pd.read_sql(sql, conn)
    df["geom"] = df["wkt_5179"].apply(_wkt.loads)
    return df


# ------------------------------------------------------------------------------------
# 그래프 구성 및 라우팅
# ------------------------------------------------------------------------------------
@dataclass
class GraphConfig:
    use_time_weight: bool = False
    default_speed_kmh: float = 40.0
    length_col: str = "length_m"
    max_spd_col: str = "max_spd"
    enable_penalty: bool = False  # True면 도로등급/차로 페널티 적용


def _road_penalty(row) -> float:
    f = 1.0
    rr = getattr(row, "road_rank", None)
    try:
        if rr is not None and not (pd.isna(rr)):
            rr = int(rr)
            if rr <= 3:
                f *= 0.85  # 상위도로 선호
            elif rr >= 7:
                f *= 1.20  # 이면도로 기피
    except Exception:
        pass
    lanes = getattr(row, "lanes", None)
    try:
        if lanes is not None and not (pd.isna(lanes)):
            lanes = float(lanes)
            if lanes >= 3:
                f *= 0.95
            elif lanes < 2:
                f *= 1.15
    except Exception:
        pass
    return f


def build_graph(links: pd.DataFrame, cfg: GraphConfig = GraphConfig()) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for r in links.itertuples(index=False):
        u, v, link_id = r.F_NODE, r.T_NODE, r.LINK_ID
        L = float(getattr(r, cfg.length_col))
        spd = getattr(r, cfg.max_spd_col, None)
        try:
            spd = float(spd) if spd is not None else cfg.default_speed_kmh
        except Exception:
            spd = cfg.default_speed_kmh
        if not (10.0 <= spd <= 110.0):
            spd = cfg.default_speed_kmh

        time_s = L / (spd * 1000.0 / 3600.0)
        base_w = time_s if cfg.use_time_weight else L
        w = base_w * (_road_penalty(r) if cfg.enable_penalty else 1.0)

        # 양방향, key=link_id
        G.add_edge(u, v, key=link_id, weight=w, length_m=L, time_s=time_s, link_id=link_id)
        G.add_edge(v, u, key=link_id, weight=w, length_m=L, time_s=time_s, link_id=link_id)
    return G


class NodeLocator:
    """노드 좌표를 링크 시/종점에서 수집 → node_id별 평균 좌표 → KDTree"""
    def __init__(self, links: pd.DataFrame):
        pairs = []
        if {"start_x", "start_y", "end_x", "end_y"}.issubset(links.columns):
            for r in links.itertuples(index=False):
                if pd.notna(r.start_x) and pd.notna(r.start_y):
                    pairs.append((r.F_NODE, float(r.start_x), float(r.start_y)))
                if pd.notna(r.end_x) and pd.notna(r.end_y):
                    pairs.append((r.T_NODE, float(r.end_x), float(r.end_y)))
        df = pd.DataFrame(pairs, columns=["node_id", "x", "y"])
        if df.empty:
            self.node_ids, self.xy, self._tree, self._fast = [], np.zeros((0, 2)), None, False
            return
        agg = df.groupby("node_id", as_index=False).mean(numeric_only=True)
        self.node_ids = agg["node_id"].tolist()
        self.xy = agg[["x", "y"]].to_numpy(dtype="float64")

        try:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self.xy) if len(self.xy) else None
            self._fast = True
        except Exception:
            self._tree = None
            self._fast = False

    def nearest_node_id(self, x: float, y: float) -> Any:
        if len(self.xy) == 0:
            raise RuntimeError("노드 좌표 없음: node 스냅 불가")
        if self._fast and self._tree is not None:
            _, idx = self._tree.query([x, y], k=1)
            return self.node_ids[int(idx)]
        idx = int(np.argmin(((self.xy[:, 0] - x) ** 2 + (self.xy[:, 1] - y) ** 2)))
        return self.node_ids[idx]


class EdgeLocator:
    """링크 선분 최근접 정사영"""
    def __init__(self, links: pd.DataFrame):
        if "geom" not in links.columns or links["geom"].isna().all():
            raise RuntimeError("geometry(WKT)가 없어 edge 스냅 불가")
        self.links = links.reset_index(drop=True)

    def nearest_edge(self, x: float, y: float) -> Tuple[int, float, Point, float]:
        pt = Point(x, y)
        best = (None, 1e18, None, None)  # (idx, dist, t_ratio, proj)
        for i, geom in enumerate(self.links["geom"]):
            if not isinstance(geom, LineString):
                continue
            t = geom.project(pt, normalized=False)
            proj = geom.interpolate(t, normalized=False)
            d = pt.distance(proj)
            if d < best[1]:
                t_ratio = float(t / geom.length) if geom.length > 0 else 0.0
                best = (i, d, t_ratio, proj)
        idx, dist, t_ratio, proj = best
        if idx is None:
            raise RuntimeError("최근접 링크 탐색 실패")
        return idx, t_ratio, proj, dist


def _insert_virtual_node(
    G: nx.MultiDiGraph,
    links_df: pd.DataFrame,
    link_row_idx: int,
    t_ratio: float,
    virt_id: Any,
    cfg: GraphConfig,
) -> Any:
    row = links_df.iloc[link_row_idx]
    u = row.F_NODE
    v = row.T_NODE
    link_id = row.LINK_ID

    L = float(getattr(row, cfg.length_col))
    spd = row.max_spd if pd.notna(row.max_spd) else cfg.default_speed_kmh
    try:
        spd = float(spd)
    except Exception:
        spd = cfg.default_speed_kmh
    if not (10.0 <= spd <= 110.0):
        spd = cfg.default_speed_kmh

    time_s = L / (spd * 1000.0 / 3600.0)
    total_w = time_s if cfg.use_time_weight else L

    L1, L2 = L * t_ratio, L * (1 - t_ratio)
    if cfg.use_time_weight:
        w1, w2 = total_w * t_ratio, total_w * (1 - t_ratio)
        t1, t2 = time_s * t_ratio, time_s * (1 - t_ratio)
    else:
        w1, w2 = L1, L2
        t1, t2 = time_s * (L1 / L if L else 0), time_s * (L2 / L if L else 0)

    # 원래 간선 제거(해당 key만)
    if G.has_edge(u, v, key=link_id):
        G.remove_edge(u, v, key=link_id)
    if G.has_edge(v, u, key=link_id):
        G.remove_edge(v, u, key=link_id)

    key_a = f"{link_id}#a"
    key_b = f"{link_id}#b"

    # u <-> virt
    G.add_edge(u, virt_id, key=key_a, weight=w1, length_m=L1, time_s=t1, link_id=link_id)
    G.add_edge(virt_id, u, key=key_a, weight=w1, length_m=L1, time_s=t1, link_id=link_id)
    # virt <-> v
    G.add_edge(virt_id, v, key=key_b, weight=w2, length_m=L2, time_s=t2, link_id=link_id)
    G.add_edge(v, virt_id, key=key_b, weight=w2, length_m=L2, time_s=t2, link_id=link_id)
    return virt_id


@dataclass
class RouteResult:
    nodes: List[Any]
    links: List[Any]
    total_length_m: float
    total_time_s: float
    total_weight: float


def _summarize(G: nx.MultiDiGraph, path: List[Any]) -> Tuple[float, float, float, List[Any]]:
    L = T = W = 0.0
    link_ids: List[Any] = []
    for a, b in zip(path[:-1], path[1:]):
        datas = G.get_edge_data(a, b)
        if not datas:
            continue
        k_best = min(datas.keys(), key=lambda k: datas[k].get("weight", 1e18))
        d = datas[k_best]
        L += float(d.get("length_m", 0.0))
        T += float(d.get("time_s", 0.0))
        W += float(d.get("weight", 0.0))
        link_ids.append(d.get("link_id"))
    return L, T, W, link_ids


class CoordinateRouter:
    def __init__(self, links_df: pd.DataFrame, cfg: GraphConfig = GraphConfig()):
        self.links = links_df
        self.cfg = cfg
        self.G = build_graph(links_df, cfg)
        self.node_locator = NodeLocator(links_df)
        self.edge_locator = EdgeLocator(links_df)
        self.to_metric = Transformer.from_crs(SRID_WGS84, SRID_METRIC, always_xy=True)

    def _nearest_edge_with_sanity(self, x: float, y: float, max_ok: float = SANITY_MAX_M):
        idx, t_ratio, proj, dist = self.edge_locator.nearest_edge(x, y)
        if dist > max_ok:
            raise RuntimeError(f"[SNAP-ERROR] nearest edge {dist:.1f} m > {max_ok} m")
        return idx, t_ratio

    def route_by_coords(
        self,
        origin_lon: float,
        origin_lat: float,
        dest_lon: float,
        dest_lat: float,
        mode: str = "edge",
        weight_key: str = "weight",
    ) -> RouteResult:
        ox, oy = _to_xy(origin_lon, origin_lat, self.to_metric)
        dx, dy = _to_xy(dest_lon, dest_lat, self.to_metric)

        G = self.G.copy()

        if mode == "edge":
            i_s, t_s = self._nearest_edge_with_sanity(ox, oy)
            sid = ("virt", "s")
            _insert_virtual_node(G, self.links, i_s, t_s, sid, self.cfg)

            i_t, t_t = self._nearest_edge_with_sanity(dx, dy)
            tid = ("virt", "t")
            _insert_virtual_node(G, self.links, i_t, t_t, tid, self.cfg)

        elif mode == "node":
            sid = self.node_locator.nearest_node_id(ox, oy)
            tid = self.node_locator.nearest_node_id(dx, dy)
        else:
            raise ValueError("mode는 'edge' 또는 'node'")

        node_path = nx.shortest_path(G, source=sid, target=tid, weight=weight_key)
        L, T, W, links = _summarize(G, node_path)
        return RouteResult(node_path, links, L, T, W)


# ------------------------------------------------------------------------------------
# 스테이션 좌표 로드 & 경로 계산
# ------------------------------------------------------------------------------------
def _ensure_station_pair(db: Session, start_station_id: int, end_station_id: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    rows = (
        db.execute(
            select(Station).where(Station.station_id.in_([start_station_id, end_station_id]))
        )
        .scalars()
        .all()
    )
    by_id = {r.station_id: r for r in rows}
    if start_station_id not in by_id or end_station_id not in by_id:
        raise HTTPException(status_code=400, detail="start or end station_id does not exist")
    s, e = by_id[start_station_id], by_id[end_station_id]
    # x=lon, y=lat
    return (float(s.x), float(s.y)), (float(e.x), float(e.y))


def compute_links_via_router(
    db: Session,
    start_station_id: int,
    end_station_id: int,
    ptype: PathType,
) -> List[int]:
    # 1) 정류장 좌표 (WGS84)
    (o_lon, o_lat), (d_lon, d_lat) = _ensure_station_pair(db, start_station_id, end_station_id)

    # 2) 로컬 BBOX로 링크 제한(성능)
    bbox = make_bbox_metric_around(o_lon, o_lat, d_lon, d_lat, buffer_m=BBOX_BUFFER_M)

    # 3) 링크 로드 및 그래프 구성
    links = load_links_from_postgis(engine, table=UROAD_TABLE, schema=None, srid_metric=SRID_METRIC, bbox_metric=bbox)

    if links.empty:
        raise HTTPException(status_code=422, detail="no road links in bbox")

    # 4) 모드별 설정
    if ptype == "shortest":
        cfg = GraphConfig(use_time_weight=False, default_speed_kmh=40.0, enable_penalty=False)
    else:  # "optimal" 또는 existing의 fallback 계산
        # 최적: 시간 가중 + 도로등급/차로 페널티
        cfg = GraphConfig(use_time_weight=True, default_speed_kmh=40.0, enable_penalty=True)

    router_algo = CoordinateRouter(links, cfg)

    try:
        res = router_algo.route_by_coords(o_lon, o_lat, d_lon, d_lat, mode="edge")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"routing failed: {e}")

    if not res.links:
        raise HTTPException(status_code=422, detail="empty link_list from router")

    # link_id가 None 포함될 수 있어 필터링 + 정수화
    out = [int(x) for x in res.links if x is not None]
    if not out:
        raise HTTPException(status_code=422, detail="no valid link ids")
    return out


# ------------------------------------------------------------------------------------
# API: 경로 생성/조회 (캐시 + 계산)
#  - existing: 있으면 그대로, 없으면 '최적' 알고리즘으로 계산 후 existing 범위에 저장
#  - shortest/optimal: 계산 후 각 범위에 저장
# ------------------------------------------------------------------------------------
@router.post("/", response_model=PathOut, status_code=status.HTTP_201_CREATED)
def create_or_get_path(payload: PathCreate, response: Response, db: Session = Depends(get_db)):
    if payload.start_station_id == payload.end_station_id:
        raise HTTPException(status_code=400, detail="start_station_id and end_station_id must differ")

    found = _find_path(db, payload.start_station_id, payload.end_station_id, payload.type)
    if found:
        response.status_code = status.HTTP_200_OK
        return PathOut(
            path_id=found.path_id,
            start_station_id=found.start_station_id,
            end_station_id=found.end_station_id,
            link_list=found.link_list,
        )

    # 계산
    if payload.type == "existing":
        # existing이 없으면 최적 경로로 계산하되, 저장은 existing 범위에
        link_list = compute_links_via_router(db, payload.start_station_id, payload.end_station_id, "optimal")  # fallback 계산
        target_type: PathType = "existing"
    elif payload.type == "shortest":
        link_list = compute_links_via_router(db, payload.start_station_id, payload.end_station_id, "shortest")
        target_type = "shortest"
    else:  # "optimal"
        link_list = compute_links_via_router(db, payload.start_station_id, payload.end_station_id, "optimal")
        target_type = "optimal"

    for _ in range(5):
        pid = _next_path_id(db, target_type)
        try:
            row = Path(
                path_id=pid,
                start_station_id=payload.start_station_id,
                end_station_id=payload.end_station_id,
                link_list=[int(x) for x in link_list],
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return PathOut(
                path_id=row.path_id,
                start_station_id=row.start_station_id,
                end_station_id=row.end_station_id,
                link_list=row.link_list,
            )
        except IntegrityError:
            db.rollback()
            continue

    raise HTTPException(status_code=409, detail="could not allocate a unique path_id")
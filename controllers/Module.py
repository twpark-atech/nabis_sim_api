import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString
from datetime import time, datetime, timedelta
from sqlalchemy import create_engine, text

import rasterio
from rasterio.sample import sample_gen
from pyproj import Transformer, Geod
from pathlib import Path


DB_URL = "postgresql://postgres:postgres@172.30.1.66:5432/new_dashboard"
engine = create_engine(DB_URL)


# ===============================
# 속도 관련 알고리즘
# ===============================
def calculate_distance_from_speed(v_start_mps, v_end_mps, dt=0.1):
    return (v_start_mps + v_end_mps) * 0.5 * dt

def estimate_distance_from_speed(v_start_kmh, v_end_kmh, dt=0.1, a_avg=1.0, b_avg=1.5, max_steps=10000):
    if abs(v_start_kmh - v_end_kmh) < 1e-12:
        return 0.0
    v_cur = v_start_kmh / 3.6
    v_target = v_end_kmh / 3.6
    v_prev = v_cur
    distance = 0.0
    for _ in range(max_steps):
        diff = v_target - v_cur
        if diff > 0:
            a_target = min(diff / dt, a_avg)
        else:
            a_target = max(diff / dt, -b_avg)
        v_next = max(0.0, v_cur + a_target * dt)
        distance += calculate_distance_from_speed(v_prev, v_next, dt)
        v_prev = v_next
        v_cur = v_next
        if abs(v_cur - v_target) < 0.05:
            break
    return distance

def _accel_step_kmh(a_ms2, dt):
    return a_ms2 * 3.6 * dt

def _integrate_dist_kmh(prev_kmh, next_kmh, dt):
    return calculate_distance_from_speed(prev_kmh/3.6, next_kmh/3.6, dt)

def simulate_speed_with_noise(v_start, v_end, v_max, end_distance, dt=0.1,
                              a_avg=1.0, b_avg=1.5, sigma=0.2, seed=None):
    """
    실제 운행 속도를 난수 기반으로 생성
    """
    rng = np.random.default_rng(seed)
    v_cur = float(v_start)
    speeds = [v_cur]

    total_dist = 0.0
    while total_dist < end_distance:
        diff = v_end - v_cur
        a_target = (min if diff > 0 else max)(
            diff / dt,
            a_avg * 3.6 if diff > 0 else -b_avg * 3.6
        )
        a_kmh_s = a_target + rng.normal(0.0, sigma * 3.6)
        v_next = max(0.0, min(v_cur + a_kmh_s * dt, v_max))
        step_dist = _integrate_dist_kmh(v_cur, v_next, dt)

        if total_dist + step_dist > end_distance:
            break

        total_dist += step_dist
        v_cur = v_next
        speeds.append(round(v_cur, 2))

    return speeds


# ===============================
# 위치 탐색 알고리즘
# ===============================
def get_nodes_with_station(station_id: int):
    query = text("""
        SELECT "x", "y"
        FROM "SIM_BIS_BUS_STATION_LOCATION"
        WHERE "station_id" = :station_id
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {"station_id": station_id}).mappings().fetchone()
    return (row["x"], row["y"]) if row else None

def get_nodes_with_xy(x, y):
    query = text("""
        SELECT "LINK_ID"
        FROM new_uroad
        ORDER BY ST_Transform(geometry, 5179) <-> ST_Transform(ST_SetSRID(ST_MakePoint(:x, :y), 4326), 5179)
        LIMIT 1;
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"x": x, "y": y})
    return df["LINK_ID"][0]

def get_nodes_with_traffic_light(link_id: int):
    query_nodes = text("""
        SELECT "F_NODE", "T_NODE"
        FROM new_uroad
        WHERE "LINK_ID" = :link_id
    """)
    coords = []
    with engine.connect() as conn:
        row = conn.execute(query_nodes, {"link_id": link_id}).mappings().fetchone()
        if not row:
            return coords
        for node_id in [row["F_NODE"], row["T_NODE"]]:
            query_light = text("""
                SELECT "NODE_ID", "NODE_LAT", "NODE_LNG"
                FROM utraffic_light_info
                WHERE "NODE_ID" = :node_id
            """)
            light_row = conn.execute(query_light, {"node_id": node_id}).mappings().fetchone()
            if light_row:
                coords.append({light_row["NODE_ID"]: [light_row["NODE_LNG"], light_row["NODE_LAT"]]})
    return coords

def compute_total_length(link_list):
    if not link_list:
        return 0.0
    query = text("""
        SELECT ST_Length(ST_Transform(geometry, 5179)) AS length_m
        FROM new_uroad
        WHERE "LINK_ID" = ANY(:link_list)
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"link_list": list(map(str, link_list))})
    return df["length_m"].sum()

def union_link_list(link_list):
    if len(link_list) < 2:
        return None
    query = text("""
        SELECT "LINK_ID", "F_NODE", "T_NODE"
        FROM new_uroad
        WHERE "LINK_ID" = ANY(:ids)
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"ids": list(map(str, link_list))}).mappings().all()
    if len(rows) < 2:
        return None
    nodes1 = {rows[0]["F_NODE"], rows[0]["T_NODE"]}
    nodes2 = {rows[1]["F_NODE"], rows[1]["T_NODE"]}
    return list(nodes1.intersection(nodes2)) if nodes1.intersection(nodes2) else None

def get_link_geometry(link_id: int):
    query = text("""
        SELECT ST_AsText(geometry) AS geom
        FROM new_uroad
        WHERE "LINK_ID" = ':link_id'
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {"link_id": link_id}).mappings().fetchone()
    return LineString(wkt.loads(row["geom"])) if row else None




def profile_to_excel(SCENARIO_ID):
    # =========================
    # 설정
    # =========================
    DB_URL = "postgresql://postgres:postgres@172.30.1.66:5432/new_dashboard"
    DT = 0.1                 # [s]
    G = 9.80665              # [m/s^2]

    # 지자기(간단 모델) – 총장/경사/편차(도)
    MAG_TOTAL_UT = 55.0      # 총 지자기 세기 (uT)
    MAG_DIP_DEG  = 55.0      # 경사(자기경사, 아래로 +)
    MAG_DEC_DEG  = 7.0       # 편차(자기편차, 동쪽 +). 한국 평균 대략 +7°

    # 래스터 파일 경로(가능한 조합을 순차 시도)
    RASTER_DIR = Path("/mnt/d/nabis/도심융합특구 관련 데이터/2025/지형경사도/DEM")
    PATH_DZDX   = RASTER_DIR / "dem_ulsan_dzdx.tif"
    PATH_DZDY   = RASTER_DIR / "dem_ulsan_dzdy.tif"
    PATH_SLOPE  = RASTER_DIR / "dem_ulsan_slope_deg.tif"
    PATH_ASPECT = RASTER_DIR / "dem_ulsan_aspect_deg.tif"
    PATH_PITCH  = RASTER_DIR / "dem_ulsan_pitch_deg.tif"
    PATH_ROLL   = RASTER_DIR / "dem_ulsan_roll_deg.tif"

    OUT_XLSX = f"scenario_{SCENARIO_ID}_profile.xlsx"

    # =========================
    # 유틸
    # =========================
    def unwrap_deg(arr_deg: np.ndarray) -> np.ndarray:
        rad = np.deg2rad(arr_deg)
        return np.unwrap(rad)

    def bearing_series_from_coords(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        geod = Geod(ellps="WGS84")
        bearings = np.zeros(len(lons), dtype=float)
        for i in range(1, len(lons)):
            az12, _, _ = geod.inv(lons[i-1], lats[i-1], lons[i], lats[i])
            bearings[i] = (az12 + 360.0) % 360.0
        if len(bearings) > 1:
            bearings[0] = bearings[1]
        return bearings

    def derivative_central(x: np.ndarray, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = x.shape[0]
        if n == 0: return np.array([], dtype=float)
        if n == 1: return np.zeros(1, dtype=float)
        dx = np.zeros_like(x)
        dx[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
        dx[0]     = (x[1] - x[0]) / dt
        dx[-1]    = (x[-1] - x[-2]) / dt
        return dx

    def sample_raster_points(raster_path: Path, xy_in_wgs84: list[tuple[float, float]]) -> np.ndarray:
        with rasterio.open(raster_path) as src:
            if src.crs is None:
                raise RuntimeError(f"CRS missing for raster: {raster_path}")
            if src.crs.to_string() != "EPSG:4326":
                to_raster = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                pts = [to_raster.transform(x, y) for (x, y) in xy_in_wgs84]
            else:
                pts = xy_in_wgs84
            return np.array([v[0] for v in sample_gen(src, pts)], dtype=float)

    def compute_pitch_roll_from_dz(dzdx: np.ndarray, dzdy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # ENU 기준: x=E, y=N, z=Up
        pitch = np.degrees(np.arctan2(-dzdx, 1.0))  # about Y
        roll  = np.degrees(np.arctan2( dzdy, 1.0))  # about X
        return pitch, roll

    def accel_body_from_long_lat_gravity(a_long: np.ndarray, a_lat: np.ndarray,
                                        pitch_deg: np.ndarray, roll_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pr = np.deg2rad(pitch_deg)
        rr = np.deg2rad(roll_deg)
        ax = a_long + G * np.sin(pr)
        ay = a_lat  - G * np.sin(rr) * np.cos(pr)
        az = - G * np.cos(pr) * np.cos(rr)
        return ax, ay, az

    def magnetometer_with_dip(yaw_deg: np.ndarray, pitch_deg: np.ndarray, roll_deg: np.ndarray,
                            B_total_uT: float, dip_deg: float, dec_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        간단 자력계: 지구자기장(ENU 고정) -> 차량 바디 프레임으로 회전.
        - dip: 아래(+), dec: 동쪽(+). 한국 평균치 근사.
        """
        # 지구고정 ENU 성분
        D = np.deg2rad(dec_deg)
        I = np.deg2rad(dip_deg)
        # 수평 성분 Bh, 수직 성분 Bv(Down +)
        Bh = B_total_uT * np.cos(I)
        Bv = B_total_uT * np.sin(I)  # Down(+)
        # ENU: (E, N, U). Down(+) 이면 U = -Bv
        B_enu = np.array([Bh*np.sin(D), Bh*np.cos(D), -Bv])  # shape (3,)

        # 바디 회전: R = Rz(yaw) * Ry(pitch) * Rx(roll)
        yr = np.deg2rad(yaw_deg)
        pr = np.deg2rad(pitch_deg)
        rr = np.deg2rad(roll_deg)

        cy, sy = np.cos(yr), np.sin(yr)
        cp, sp = np.cos(pr), np.sin(pr)
        cr, sr = np.cos(rr), np.sin(rr)

        # 벡터 반복 적용
        mx = cy*cp*B_enu[0] + (cy*sp*sr - sy*cr)*B_enu[1] + (cy*sp*cr + sy*sr)*B_enu[2]
        my = sy*cp*B_enu[0] + (sy*sp*sr + cy*cr)*B_enu[1] + (sy*sp*cr - cy*sr)*B_enu[2]
        mz =      -sp*B_enu[0] +            cp*sr*B_enu[1] +            cp*cr*B_enu[2]
        return mx, my, mz

    def format_time_strings(start_time_obj, n, dt) -> list[str]:
        """
        Time: 'HH:MM:SS.S' 형식 문자열 (예: 01:35:06.4)
        start_time_obj 기준으로 시작.
        """
        if isinstance(start_time_obj, datetime):
            base_dt = start_time_obj
        elif isinstance(start_time_obj, time):
            base_dt = datetime.combine(datetime(2000,1,1), start_time_obj)
        else:
            raise ValueError("start_time_obj must be datetime or time")

        times = []
        for i in range(n):
            cur = base_dt + timedelta(seconds=(i*dt))
            # 시:분:초.1 (소수 1자리)
            s = f"{cur.hour:02d}:{cur.minute:02d}:{cur.second:02d}.{int(cur.microsecond/100000)}"
            times.append(s)
        return times



    # =========================
    # 1) DB에서 시나리오 로드
    # =========================
    engine = create_engine(DB_URL)
    row = pd.read_sql(text("""
        SELECT start_time, speed_list, coord_list
        FROM "SIM_SCENARIOS"
        WHERE scenario_id = :sid
    """), engine, params={"sid": SCENARIO_ID}).iloc[0]

    start_time = row["start_time"]
    speed_list = row["speed_list"]
    coord_list = row["coord_list"]

    # =========================
    # 2) 시계열/기본량
    # =========================
    N = len(speed_list)
    t_sec = np.arange(N, dtype=float) * DT

    wheel_mps  = np.asarray(speed_list, dtype=float)       # m/s
    wheel_kmh  = wheel_mps * 3.6                           # km/h

    lons = np.array([pt[0] for pt in coord_list], dtype=float)
    lats = np.array([pt[1] for pt in coord_list], dtype=float)
    if len(lons) != N:
        M = min(N, len(lons))
        t_sec     = t_sec[:M]
        wheel_kmh = wheel_kmh[:M]
        lons      = lons[:M]
        lats      = lats[:M]
        N = M

    # =========================
    # 3) 지형 기반 pitch/roll (우선순위: pitch/roll tif > dzdx/dzdy > slope/aspect)
    # =========================
    xy_wgs84 = list(zip(lons, lats))
    has_pitch_roll   = PATH_PITCH.exists() and PATH_ROLL.exists()
    has_dz           = PATH_DZDX.exists() and PATH_DZDY.exists()
    has_slope_aspect = PATH_SLOPE.exists() and PATH_ASPECT.exists()

    if has_pitch_roll:
        with rasterio.open(PATH_PITCH) as _:
            pass
        pitch_deg = sample_raster_points(PATH_PITCH, xy_wgs84)
        roll_deg  = sample_raster_points(PATH_ROLL,  xy_wgs84)
    elif has_dz:
        dzdx = sample_raster_points(PATH_DZDX, xy_wgs84)
        dzdy = sample_raster_points(PATH_DZDY, xy_wgs84)
        pitch_deg, roll_deg = compute_pitch_roll_from_dz(dzdx, dzdy)
    elif has_slope_aspect:
        slope_deg  = sample_raster_points(PATH_SLOPE,  xy_wgs84)
        aspect_deg = sample_raster_points(PATH_ASPECT, xy_wgs84)
        m  = np.tan(np.deg2rad(slope_deg))
        az = np.deg2rad(aspect_deg)
        dzdx = m * np.sin(az)
        dzdy = m * np.cos(az)
        pitch_deg, roll_deg = compute_pitch_roll_from_dz(dzdx, dzdy)
    else:
        raise FileNotFoundError("Need one of: (pitch&roll) or (dzdx&dzdy) or (slope&aspect) rasters.")

    # =========================
    # 4) yaw/가속도/자이로/자력계
    # =========================
    yaw_deg = bearing_series_from_coords(lons, lats)        # 0~360
    yaw_rad_unwrap = unwrap_deg(yaw_deg)                    # rad (연속)
    yaw_rate_rad_s = derivative_central(yaw_rad_unwrap, DT)

    a_long = derivative_central(wheel_kmh/3.6, DT)          # m/s^2
    a_lat  = (wheel_kmh/3.6) * yaw_rate_rad_s               # m/s^2

    # accel (body): m/s^2 → g
    acc_x_ms2, acc_y_ms2, acc_z_ms2 = accel_body_from_long_lat_gravity(a_long, a_lat, pitch_deg, roll_deg)
    acc_x_g = acc_x_ms2 / G
    acc_y_g = acc_y_ms2 / G
    acc_z_g = acc_z_ms2 / G

    # gyro: rad/s → deg/s
    gyro_x_deg_s = np.degrees(derivative_central(np.deg2rad(roll_deg),  DT))
    gyro_y_deg_s = np.degrees(derivative_central(np.deg2rad(pitch_deg), DT))
    gyro_z_deg_s = np.degrees(yaw_rate_rad_s)

    # magnetometer: uT
    mag_x_uT, mag_y_uT, mag_z_uT = magnetometer_with_dip(
        yaw_deg, pitch_deg, roll_deg,
        B_total_uT=MAG_TOTAL_UT, dip_deg=MAG_DIP_DEG, dec_deg=MAG_DEC_DEG
    )

    # =========================
    # 5) Time 문자열(예: 01:35.6)
    # =========================
    time_str = format_time_strings(start_time, N, DT)

    # =========================
    # 6) DataFrame & Excel 저장 (단위 2행)
    # =========================
    cols = ["Time","WheelSpeed","acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z",
            "mag_x","mag_y","mag_z","pitch","roll","yaw"]

    units = ["","km/h","g","g","g","deg/s","deg/s","deg/s","uT","uT","uT","deg","deg","deg"]

    data = pd.DataFrame({
        "Time": time_str,
        "WheelSpeed": wheel_kmh,
        "acc_x": acc_x_g, "acc_y": acc_y_g, "acc_z": acc_z_g,
        "gyro_x": gyro_x_deg_s, "gyro_y": gyro_y_deg_s, "gyro_z": gyro_z_deg_s,
        "mag_x": mag_x_uT, "mag_y": mag_y_uT, "mag_z": mag_z_uT,
        "pitch": pitch_deg, "roll": roll_deg, "yaw": yaw_deg
    })

    # 단위 행을 첫 번째 데이터 행으로 삽입
    units_row = pd.DataFrame([dict(zip(cols, units))])
    out_df = pd.concat([units_row, data], ignore_index=True)
    out_df = out_df[cols]

    return out_df
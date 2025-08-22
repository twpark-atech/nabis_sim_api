# 도심융합특구 시뮬레이션 모델 API
## Project
### Workplace Structure
```
daeduk_api
├─app
│  └─ api
├─controllers
├─bus_oppor_induce_prediction # 추가 예정
└─traffic_congest_prediction # 추가 예정
```
### 구축 환경
|                | version |
| ------------------------- | --------------- |
| Platform | AMD64 |
| Python | 3.10.18 |
| Ubuntu | 22.04 |
| CPU | i9-12900K |
| RAM | 64GB |
| GPU | RTX 3090 |

### 실행 방법
1. 가상 환경 생성
```
conda create -n nabis_sim python==3.10
conda activate nabis_sim
```

2. 필수 패키지 설치
```
pip install -r requirements.txt
```

3. uvicorn 실행
```
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. API 사용
- /api/v1/stations/
    - Input: {"station_name": "string", "x": float, "y": float}
    - Output: {"station_id": int, "station_name": "string", "x": float, "y": float}
- /api/v1/routes/
    - Input: {"route_name": "string", "start_station_id": int, "end_station_id": int, "station_list": [int]}
    - Output: {"route_id": int, "route_name": "string", "start_station_id": int, "end_station_id": int, "station_list": [int]}
- /api/v1/paths/
    - Input: {"start_station_id": int, "end_station_id": int, "type": "string"}
    - Output: {"path_id": int, "start_station_id": int, "end_station_id": int, "link_list": [int]}
    - Type: "existing", "shortest", "optimal" # existing은 참고용.
- /api/v1/scenarios/
    - Input: {"name": "string", "route_id": int, "headway_min": int, "start_time": time, "end_time": time, "departure_time": time, "path_type": "string"}
    - Output: {"scenario_id": int, "name": "string", "route_id": int, "headway_min": int, "start_time": time, "end_time": time, "departure_time": time, "path_type": "string", "route_length": float, "route_curvature": float, "speed_list": [float], "coord_list": [[float, float]], "link_list": [int]}
    - Type: "existing", "shortest", "optimal" # existing은 참고용.
- /api/v1/profile/{scenario_id}/excel
    - Input: {"scenario_id": int}
    - Output: BLOB_URL
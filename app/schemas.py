# app/schemas.py
from typing import Optional, List, Dict, Literal, Tuple
from pydantic import BaseModel, Field, confloat, conint, ConfigDict, AliasChoices
from datetime import time


# 정류장
class StationBase(BaseModel):
    station_name: str = Field(
        ...,
        max_length=200, 
        validation_alias=AliasChoices("station_name", "name"),
        serialization_alias="station_name",
    )
    lat: confloat(ge=-90, le=90)
    lon: confloat(ge=-180, le=180)

class StationCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    station_name: str = Field(
        ...,
        max_length=200,
        validation_alias=AliasChoices("station_name", "name"),
        serialization_alias="station_name",
    )
    x: float
    y: float

class StationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    station_id: int
    station_name: str
    x: float
    y: float

# 노선
class RouteCreate(BaseModel):
    route_name: str = Field(
        ...,
        max_length=200,    
    )
    start_station_id: int
    end_station_id: int
    station_list: List[int]

class RouteOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    route_id: int
    route_name: str
    start_station_id: int
    end_station_id: int
    station_list: List[int]

# 경로
PathType = Literal["existing", "shortest", "optimal"]

class PathCreate(BaseModel):
    start_station_id: int
    end_station_id: int
    type: PathType

class PathOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    path_id: int
    start_station_id: int
    end_station_id: int
    link_list: list[int]

# 시나리오
class ScenarioCreate(BaseModel):
    name: str = Field(..., max_length=200)
    route_id: int
    headway_min: int
    start_time: time
    end_time: time
    departure_time: time
    path_type: PathType
    existing_list: List[int]

class ScenarioOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    scenario_id: int
    name: str
    route_id: int

    headway_min: int
    start_time: time
    end_time: time
    departure_time: time
    path_type: PathType

    route_length: float
    route_curvature: float

    speed_list: List[float]
    coord_list: List[Tuple[float, float]]
    link_list: List[int]
    status: str
    existing_list: List[int]
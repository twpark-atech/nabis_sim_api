# app/schemas.py
from typing import Optional, List, Dict, Literal
from pydantic import BaseModel, Field, confloat, conint, ConfigDict, AliasChoices

# ---------- Station ----------
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

# ---------- Route ----------
class StationInfo(BaseModel):
    station_id: int
    station_name: str
    x: float
    y: float

class RouteCreate(BaseModel):
    route_name: str = Field(..., max_length=200)
    station_list: List[int]

class RouteOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    route_id: int
    route_name: str
    station_list: List[int]

class RouteStationsOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    route_id: int
    route_name: str
    stations: List[StationInfo]

# ---------- Path ----------
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

# ---------- Scenario ----------
class ScenarioOut(BaseModel):
    scenario_id: int
    name: str
    params_json: str
    class Config:
        from_attributes = True

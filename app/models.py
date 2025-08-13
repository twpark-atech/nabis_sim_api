# app/models.py
from datetime import datetime
from sqlalchemy import (
    BigInteger, Integer, String, Text, DateTime, ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import ARRAY

class Base(DeclarativeBase):
    pass

# 정류장
class Station(Base):
    __tablename__ = "new_bis_bus_station_location"
    station_id: Mapped[int]   = mapped_column(BigInteger, primary_key=True)
    station_name: Mapped[str] = mapped_column(String(200), nullable=False)
    x: Mapped[float]          = mapped_column(nullable=False)  # lon EPSG:4326
    y: Mapped[float]          = mapped_column(nullable=False)  # lat  EPSG:4326
    __table_args__ = (Index("ix_station_xy", "x", "y"),)

# 노선
class Route(Base):
    __tablename__ = "new_routes"
    route_id: Mapped[int]         = mapped_column(BigInteger, primary_key=True)  # CSV의 노선ID 그대로
    route_name: Mapped[str]       = mapped_column(String(200), nullable=False)
    station_list: Mapped[list[int]] = mapped_column(ARRAY(BigInteger), nullable=False)

# 경로(정류장-정류장)
class Path(Base):
    __tablename__ = "new_paths"
    path_id: Mapped[int]            = mapped_column(BigInteger, primary_key=True)  # 6억 번대
    start_station_id: Mapped[int]   = mapped_column(BigInteger, nullable=False)
    end_station_id: Mapped[int]     = mapped_column(BigInteger, nullable=False)
    link_list: Mapped[list[int]]    = mapped_column(ARRAY(BigInteger), nullable=False)

    __table_args__ = (
        Index("ix_paths_pair", "start_station_id", "end_station_id"),
    )

# 시나리오(간단 메타)
class Scenario(Base):
    __tablename__ = "scenarios"
    scenario_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str]        = mapped_column(String(200))
    # 입력 파라미터(배차, 운행시작/종료, 출발시간 등)를 JSON으로 보관
    params_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

# app/models.py
from datetime import time
from sqlalchemy import BigInteger, String, Text, Time, Index, Float, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, ENUM as PG_ENUM

# 베이스
class Base(DeclarativeBase):
    pass

# 정류장
class Station(Base):
    __tablename__ = "SIM_BIS_BUS_STATION_LOCATION"
    station_id: Mapped[int]         = mapped_column(BigInteger, primary_key=True)
    station_name: Mapped[str]       = mapped_column(String(200), nullable=False)
    x: Mapped[float]                = mapped_column(nullable=False)
    y: Mapped[float]                = mapped_column(nullable=False)
    __table_args__ = (Index("ix_station_xy", "x", "y"),)

# 노선
class Route(Base):
    __tablename__ = "SIM_ROUTES"
    route_id: Mapped[int]           = mapped_column(BigInteger, primary_key=True)
    route_name: Mapped[str]         = mapped_column(String(200), nullable=False)
    start_station_id: Mapped[int]   = mapped_column(BigInteger, nullable=False)
    end_station_id: Mapped[int]     = mapped_column(BigInteger, nullable=False)
    station_list: Mapped[list[int]] = mapped_column(ARRAY(BigInteger), nullable=False)

# 경로
class Path(Base):
    __tablename__ = "SIM_PATHS"
    path_id: Mapped[int]            = mapped_column(BigInteger, primary_key=True)
    start_station_id: Mapped[int]   = mapped_column(BigInteger, nullable=False)
    end_station_id: Mapped[int]     = mapped_column(BigInteger, nullable=False)
    link_list: Mapped[list[int]]    = mapped_column(ARRAY(BigInteger), nullable=False)
    __table_args__ = (Index("ix_paths_pair", "start_station_id", "end_station_id"),)

# 시나리오
PATH_TYPE_ENUM = PG_ENUM("existing", "shortest", "optimal", name="path_type_enum", create_type=False)

class Scenario(Base):
    __tablename__ = "SIM_SCENARIOS"

    # 시나리오 ID는 클라이언트에서 제공(autoincrement 아님)
    scenario_id: Mapped[int]        = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str]               = mapped_column(String(200), nullable=False)

    route_id: Mapped[int]           = mapped_column(BigInteger, nullable=False)
    headway_min: Mapped[int]        = mapped_column(Integer, nullable=False)
    start_time: Mapped[time]        = mapped_column(Time(timezone=False), nullable=False)
    end_time: Mapped[time]          = mapped_column(Time(timezone=False), nullable=False)
    departure_time: Mapped[time]    = mapped_column(Time(timezone=False), nullable=False)
    path_type: Mapped[str]          = mapped_column(PATH_TYPE_ENUM, nullable=False)

    # 🔽 Celery가 채우기 전까지는 비워둘 수 있게 모두 NULL 허용
    link_list: Mapped[list[int] | None]    = mapped_column(ARRAY(BigInteger), nullable=True)
    route_length: Mapped[float | None]     = mapped_column(Float, nullable=True)
    route_curvature: Mapped[float | None]  = mapped_column(Float, nullable=True)
    speed_list: Mapped[list[float] | None] = mapped_column(ARRAY(Float), nullable=True)
    coord_list: Mapped[list | None]        = mapped_column(JSONB, nullable=True)
    existing_list: Mapped[list[int] | None]= mapped_column(ARRAY(BigInteger), nullable=True)

    # ✨ 상태 컬럼: "생성 중" → "생성 완료"/"생성 실패"
    status: Mapped[str]             = mapped_column(String(16), nullable=False, default="생성 중", server_default="생성 중")

    __table_args__ = (
        Index("ix_sim_scenarios_route_id", "route_id"),
        Index("ix_sim_scenarios_name", "name"),
    )
from datetime import datetime
from sqlalchemy import (String, Float, Boolean, DateTime, ForeignKey,
                        UniqueConstraint, CheckConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db import Base

class Substation(Base):
    __tablename__ = "substations"
    __table_args__ = (CheckConstraint("max_capacity_mw > 0", name="ck_positive_capacity"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(160), unique=True)
    district: Mapped[str] = mapped_column(String(80), index=True)
    max_capacity_mw: Mapped[float] = mapped_column(Float)
    # Demo equipment is not a register of actual substations in Bishkek.
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    loads: Mapped[list["SubstationLoad"]] = relationship(back_populates="substation")

class SubstationLoad(Base):
    __tablename__ = "substation_loads"
    __table_args__ = (
        UniqueConstraint("substation_id", "timestamp", name="uq_station_hour"),
        CheckConstraint("actual_load_mw >= 0", name="ck_nonnegative_load"),
        CheckConstraint("humidity_pct BETWEEN 0 AND 100", name="ck_humidity"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    substation_id: Mapped[int] = mapped_column(ForeignKey("substations.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    actual_load_mw: Mapped[float] = mapped_column(Float)
    is_overloaded: Mapped[bool] = mapped_column(Boolean)
    # Feature snapshots prevent ambiguous joins across forecast runs.
    temperature_c: Mapped[float] = mapped_column(Float)
    humidity_pct: Mapped[float] = mapped_column(Float)
    weather_source: Mapped[str] = mapped_column(String(40))
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    dataset_version: Mapped[str] = mapped_column(String(60), default="unknown")
    substation: Mapped[Substation] = relationship(back_populates="loads")

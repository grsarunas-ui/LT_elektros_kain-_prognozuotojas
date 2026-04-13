from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.db import Base


class DataVersion(Base):
    __tablename__ = "data_versions"

    id = Column(Integer, primary_key=True, index=True)
    version_name = Column(String, unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=True)

    market_rows = relationship(
        "MarketData",
        back_populates="data_version",
        cascade="all, delete-orphan",
    )
    predictions = relationship(
        "Prediction",
        back_populates="data_version",
        cascade="all, delete-orphan",
    )
    metrics = relationship(
        "ModelMetric",
        back_populates="data_version",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<DataVersion(id={self.id}, version_name='{self.version_name}')>"


class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)

    data_version_id = Column(
        Integer,
        ForeignKey("data_versions.id"),
        nullable=False,
        index=True,
    )
    frequency = Column(String, nullable=False, index=True)  # 'hourly' arba '15min'
    datetime = Column(DateTime, nullable=False, index=True)

    price = Column(Float, nullable=True)

    # Nord Pool
    lv_price = Column(Float, nullable=True)
    ee_price = Column(Float, nullable=True)
    se4_price = Column(Float, nullable=True)
    pl_price = Column(Float, nullable=True)

    # Litgrid
    consumption_mw = Column(Float, nullable=True)
    production_total_mw = Column(Float, nullable=True)

    # Commercial flows
    flow_lt_lv = Column(Float, nullable=True)
    flow_lt_se = Column(Float, nullable=True)
    flow_lt_pl = Column(Float, nullable=True)
    flow_total = Column(Float, nullable=True)
    flow_abs_total = Column(Float, nullable=True)

    data_version = relationship("DataVersion", back_populates="market_rows")

    __table_args__ = (
        UniqueConstraint(
            "data_version_id",
            "frequency",
            "datetime",
            name="uq_market_data_version_frequency_datetime",
        ),
    )

    def __repr__(self):
        return (
            f"<MarketData(id={self.id}, frequency='{self.frequency}', "
            f"datetime='{self.datetime}')>"
        )


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    data_version_id = Column(
        Integer,
        ForeignKey("data_versions.id"),
        nullable=False,
        index=True,
    )
    dataset_name = Column(String, nullable=False, index=True)   # pvz. hourly_extended
    model_name = Column(String, nullable=False, index=True)     # pvz. XGBoost / MLP / LSTM
    datetime = Column(DateTime, nullable=False, index=True)

    actual_price = Column(Float, nullable=True)
    predicted_price = Column(Float, nullable=True)
    abs_error = Column(Float, nullable=True)
    error = Column(Float, nullable=True)

    data_version = relationship("DataVersion", back_populates="predictions")

    __table_args__ = (
        UniqueConstraint(
            "data_version_id",
            "dataset_name",
            "model_name",
            "datetime",
            name="uq_predictions_version_dataset_model_datetime",
        ),
    )

    def __repr__(self):
        return (
            f"<Prediction(id={self.id}, dataset='{self.dataset_name}', "
            f"model='{self.model_name}', datetime='{self.datetime}')>"
        )


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)

    data_version_id = Column(
        Integer,
        ForeignKey("data_versions.id"),
        nullable=False,
        index=True,
    )
    dataset_name = Column(String, nullable=False, index=True)
    model_name = Column(String, nullable=False, index=True)  # XGBoost / MLP / LSTM

    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    r2 = Column(Float, nullable=True)
    smape = Column(Float, nullable=True)

    data_version = relationship("DataVersion", back_populates="metrics")

    __table_args__ = (
        UniqueConstraint(
            "data_version_id",
            "dataset_name",
            "model_name",
            name="uq_metrics_version_dataset_model",
        ),
    )

    def __repr__(self):
        return (
            f"<ModelMetric(id={self.id}, dataset='{self.dataset_name}', "
            f"model='{self.model_name}')>"
        )
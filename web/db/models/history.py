from sqlalchemy import Boolean, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, relationship

from ..base import Base


class PredictRecord(Base):
    __tablename__: str = "predictrecord"

    id: Mapped[int] = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    type: Mapped[str] = Column(String, nullable=False)
    name: Mapped[str] = Column(String, nullable=False)
    result_markdown: Mapped[str] = Column(String, nullable=False)
    has_output_file = Column(Boolean, default=False)
    user_data_path: Mapped[str] = Column(String)
    output_filename: Mapped[str] = Column(String)

    user_id: Mapped[int] = Column(Integer, ForeignKey('users.id'), nullable=False)
    user = relationship("User", back_populates="predictions")
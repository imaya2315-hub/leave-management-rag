from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.core.database import Base


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    manager_id = Column(Integer, ForeignKey("employees.id"), unique=True, nullable=False)

    manager = relationship(
        "Employee",
        foreign_keys=[manager_id],
        back_populates="managed_team",
        uselist=False,
    )
    members = relationship(
        "Employee",
        foreign_keys="Employee.team_id",
        back_populates="team",
    )

from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.orm import relationship

from app.core.database import Base


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, index=True)
    last_name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    department = Column(String)
    is_active = Column(Boolean, default=True)
    role = Column(String, default="employee")  # "employee", "manager", or "admin"

    annual_leave_balance = Column(Integer, default=20)
    sick_leave_balance = Column(Integer, default=10)
    casual_leave_balance = Column(Integer, default=5)

    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True, index=True)

    leaves = relationship("Leave", back_populates="owner")
    team = relationship(
        "Team",
        foreign_keys=[team_id],
        back_populates="members",
    )
    managed_team = relationship(
        "Team",
        foreign_keys="Team.manager_id",
        back_populates="manager",
        uselist=False,
    )

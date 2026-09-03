from sqlalchemy import Column, Integer, String, Boolean
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
    role = Column(String, default="employee")  # "employee" or "manager"

    annual_leave_balance = Column(Integer, default=20)
    sick_leave_balance = Column(Integer, default=10)
    casual_leave_balance = Column(Integer, default=5)

    leaves = relationship("Leave", back_populates="owner")

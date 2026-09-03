from sqlalchemy import Column, Integer, String, Date, ForeignKey
from sqlalchemy.orm import relationship

from app.core.database import Base


class Leave(Base):
    __tablename__ = "leaves"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"))
    leave_type = Column(String, index=True)  # "Annual", "Sick", "Casual"
    start_date = Column(Date)
    end_date = Column(Date)
    status = Column(String, default="Pending")

    owner = relationship("Employee", back_populates="leaves")

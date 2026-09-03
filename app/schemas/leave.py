from datetime import date

from pydantic import BaseModel, ConfigDict


class LeaveBase(BaseModel):
    leave_type: str  # "Annual", "Sick", "Casual"
    start_date: date
    end_date: date


class LeaveCreate(LeaveBase):
    pass


class Leave(LeaveBase):
    id: int
    employee_id: int
    status: str

    model_config = ConfigDict(from_attributes=True)

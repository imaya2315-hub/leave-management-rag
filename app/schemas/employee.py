from typing import List, Literal

from pydantic import BaseModel, ConfigDict

from app.schemas.leave import Leave


class EmployeeBase(BaseModel):
    first_name: str
    last_name: str
    email: str
    department: str


class EmployeeCreate(EmployeeBase):
    password: str
    role: Literal["employee", "manager"] = "employee"


class Employee(EmployeeBase):
    id: int
    is_active: bool
    role: str
    annual_leave_balance: int
    sick_leave_balance: int
    casual_leave_balance: int
    leaves: List[Leave] = []

    model_config = ConfigDict(from_attributes=True)


class EmployeeSummary(EmployeeBase):
    id: int
    is_active: bool
    role: str
    annual_leave_balance: int
    sick_leave_balance: int
    casual_leave_balance: int

    model_config = ConfigDict(from_attributes=True)

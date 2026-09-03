"""
Business logic for employee operations. Routes call into this layer
instead of talking to crud/ or the database directly — this is where
rules like "email must be unique" live.
"""
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app import crud
from app.core.security import get_password_hash
from app.schemas.employee import EmployeeCreate


def register_employee(db: Session, employee: EmployeeCreate):
    existing = crud.employee.get_employee_by_email(db, employee.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pw = get_password_hash(employee.password)
    return crud.employee.create_employee(db, employee, hashed_pw)


def list_employees(db: Session, skip: int = 0, limit: int = 100):
    return crud.employee.get_employees(db, skip, limit)


def get_employee_or_404(db: Session, employee_id: int):
    employee = crud.employee.get_employee(db, employee_id)
    if employee is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    return employee


def update_employee(db: Session, employee_id: int, employee_update: EmployeeCreate):
    db_employee = get_employee_or_404(db, employee_id)
    return crud.employee.update_employee(db, db_employee, employee_update)


def delete_employee(db: Session, employee_id: int):
    db_employee = get_employee_or_404(db, employee_id)
    crud.employee.delete_employee(db, db_employee)
    return {"message": f"Employee {employee_id} deleted successfully"}

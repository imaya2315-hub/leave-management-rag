"""
CRUD layer: direct database operations only. No business rules here —
that belongs in services/. This layer just knows how to read/write
Employee rows.
"""
from sqlalchemy.orm import Session

from app.models.employee import Employee
from app.schemas.employee import EmployeeCreate


def get_employee_by_email(db: Session, email: str) -> Employee | None:
    return db.query(Employee).filter(Employee.email == email).first()


def get_employee(db: Session, employee_id: int) -> Employee | None:
    return db.query(Employee).filter(Employee.id == employee_id).first()


def get_employees(db: Session, skip: int = 0, limit: int = 100) -> list[Employee]:
    return db.query(Employee).offset(skip).limit(limit).all()


def create_employee(db: Session, employee: EmployeeCreate, hashed_password: str) -> Employee:
    new_employee = Employee(
        first_name=employee.first_name,
        last_name=employee.last_name,
        email=employee.email,
        department=employee.department,
        hashed_password=hashed_password,
        role=employee.role,
    )
    db.add(new_employee)
    db.commit()
    db.refresh(new_employee)
    return new_employee


def update_employee(db: Session, db_employee: Employee, employee_update: EmployeeCreate) -> Employee:
    db_employee.first_name = employee_update.first_name
    db_employee.last_name = employee_update.last_name
    db_employee.email = employee_update.email
    db_employee.department = employee_update.department
    db_employee.role = employee_update.role
    db.commit()
    db.refresh(db_employee)
    return db_employee


def delete_employee(db: Session, db_employee: Employee) -> None:
    db.delete(db_employee)
    db.commit()

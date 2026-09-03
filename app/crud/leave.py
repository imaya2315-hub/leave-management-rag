from sqlalchemy.orm import Session

from app.models.leave import Leave
from app.schemas.leave import LeaveCreate


def get_leave(db: Session, leave_id: int) -> Leave | None:
    return db.query(Leave).filter(Leave.id == leave_id).first()


def get_leaves_by_employee(db: Session, employee_id: int) -> list[Leave]:
    return db.query(Leave).filter(Leave.employee_id == employee_id).all()


def create_leave(db: Session, employee_id: int, leave_request: LeaveCreate) -> Leave:
    new_leave = Leave(
        employee_id=employee_id,
        leave_type=leave_request.leave_type,
        start_date=leave_request.start_date,
        end_date=leave_request.end_date,
        status="Pending",
    )
    db.add(new_leave)
    db.commit()
    db.refresh(new_leave)
    return new_leave


def set_leave_status(db: Session, leave: Leave, status: str) -> Leave:
    leave.status = status
    db.commit()
    db.refresh(leave)
    return leave


def cancel_leave(db: Session, leave: Leave) -> Leave:
    return set_leave_status(db, leave, "Cancelled")

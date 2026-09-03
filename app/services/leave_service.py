"""
Business logic for leave requests.

Workflow: apply (Pending, no balance change) -> manager approves
(balance deducted) or rejects (no balance change) -> employee can
cancel an Approved leave (balance refunded) or a still-Pending one
(nothing to refund, since nothing was deducted yet).
"""
from datetime import date

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app import crud
from app.models.leave import Leave as LeaveModel
from app.schemas.leave import LeaveCreate
from app.utils.helpers import calculate_working_days

LEAVE_BALANCE_FIELDS = {
    "Annual": "annual_leave_balance",
    "Sick": "sick_leave_balance",
    "Casual": "casual_leave_balance",
}


def _validate_leave(employee, leave_type: str, start_date: date, end_date: date) -> tuple[str, int]:
    """Runs all validation and returns (balance_field, requested_days) if valid."""
    if start_date < date.today():
        raise HTTPException(status_code=400, detail="Cannot apply for leave in the past")

    if end_date < start_date:
        raise HTTPException(status_code=400, detail="End date must be after start date")

    requested_days = calculate_working_days(start_date, end_date)
    if requested_days == 0:
        raise HTTPException(status_code=400, detail="Requested period contains only weekends or holidays")

    balance_field = LEAVE_BALANCE_FIELDS.get(leave_type)
    if balance_field is None:
        raise HTTPException(status_code=400, detail="Invalid leave type. Use Annual, Sick, or Casual")

    current_balance = getattr(employee, balance_field)
    if current_balance < requested_days:
        raise HTTPException(status_code=400, detail=f"Insufficient {leave_type} leave balance")

    return balance_field, requested_days


def apply_leave(db: Session, employee, leave_request: LeaveCreate):
    """
    Creates a Pending leave request for the given (already-authenticated)
    employee. Balance is only checked here for early feedback — it isn't
    deducted until a manager approves it.
    """
    _validate_leave(employee, leave_request.leave_type, leave_request.start_date, leave_request.end_date)

    # Prevent accidental duplicate pending requests. This is especially
    # important because managers identify requests by employee + dates +
    # leave type in the conversational UI.
    duplicate = (
        db.query(LeaveModel)
        .filter(
            LeaveModel.employee_id == employee.id,
            LeaveModel.leave_type == leave_request.leave_type,
            LeaveModel.start_date == leave_request.start_date,
            LeaveModel.end_date == leave_request.end_date,
            LeaveModel.status == "Pending",
        )
        .first()
    )

    if duplicate:
        raise HTTPException(
            status_code=400,
            detail="A pending leave request already exists for these dates and leave type.",
        )

    return crud.leave.create_leave(db, employee.id, leave_request)


def approve_leave(db: Session, leave_id: int):
    """Manager-only: approves a Pending leave and deducts the balance."""
    leave = crud.leave.get_leave(db, leave_id)
    if not leave:
        raise HTTPException(status_code=404, detail="Leave request not found")

    if leave.status != "Pending":
        raise HTTPException(status_code=400, detail=f"Leave is already {leave.status}, cannot approve")

    employee = crud.employee.get_employee(db, leave.employee_id)

    # Re-validate at approval time — balance may have changed since the
    # request was submitted.
    balance_field, requested_days = _validate_leave(
        employee, leave.leave_type, leave.start_date, leave.end_date
    )

    setattr(employee, balance_field, getattr(employee, balance_field) - requested_days)
    db.commit()

    return crud.leave.set_leave_status(db, leave, "Approved")


def reject_leave(db: Session, leave_id: int):
    """Manager-only: rejects a Pending leave. No balance change."""
    leave = crud.leave.get_leave(db, leave_id)
    if not leave:
        raise HTTPException(status_code=404, detail="Leave request not found")

    if leave.status != "Pending":
        raise HTTPException(status_code=400, detail=f"Leave is already {leave.status}, cannot reject")

    return crud.leave.set_leave_status(db, leave, "Rejected")


def cancel_leave(db: Session, leave_id: int):
    """
    Employee-facing cancellation. Refunds the balance only if the leave
    had already been Approved (and therefore already deducted).
    """
    leave = crud.leave.get_leave(db, leave_id)
    if not leave:
        raise HTTPException(status_code=404, detail="Leave request not found")

    if leave.status in ("Cancelled", "Rejected"):
        raise HTTPException(status_code=400, detail=f"This leave is already {leave.status.lower()}")

    if leave.status == "Approved":
        employee = crud.employee.get_employee(db, leave.employee_id)
        refund_days = calculate_working_days(leave.start_date, leave.end_date)
        balance_field = LEAVE_BALANCE_FIELDS.get(leave.leave_type)
        if balance_field:
            setattr(employee, balance_field, getattr(employee, balance_field) + refund_days)
        db.commit()

    return crud.leave.cancel_leave(db, leave)


def check_eligibility(db: Session, employee, leave_type: str, start_date: date, end_date: date) -> dict:
    """
    Non-mutating dry run of the same validation apply_leave/approve_leave
    use. Built for the AI assistant: it needs to answer "can I take N
    days off" without actually creating a leave request. Never writes
    to the database and never changes employee balances.
    """
    try:
        balance_field, requested_days = _validate_leave(employee, leave_type, start_date, end_date)
    except HTTPException as exc:
        return {
            "eligible": False,
            "reason": exc.detail,
            "requested_days": None,
            "remaining_balance_after": None,
        }

    remaining_after = getattr(employee, balance_field) - requested_days
    return {
        "eligible": True,
        "reason": "Sufficient balance and a valid date range.",
        "requested_days": requested_days,
        "remaining_balance_after": remaining_after,
    }


def get_leave_history(db: Session, employee_id: int):
    employee = crud.employee.get_employee(db, employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    return crud.leave.get_leaves_by_employee(db, employee_id)


def get_my_leave_history(db: Session, employee):
    """For the /leaves/me/ endpoint — employee is already authenticated,
    no lookup or existence check needed."""
    return crud.leave.get_leaves_by_employee(db, employee.id)

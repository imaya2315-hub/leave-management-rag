from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_manager
from app.models.employee import Employee
from app.models.leave import Leave as LeaveModel
from app.schemas.leave import Leave as LeaveSchema, LeaveCreate
from app.services import leave_service

router = APIRouter(tags=["Leaves"])


@router.post("/leaves/", response_model=LeaveSchema)
def apply_leave(
    leave_request: LeaveCreate,
    db: Session = Depends(get_db),
    current_user: Employee = Depends(get_current_user),
):
    """Apply for leave as the currently authenticated employee."""
    return leave_service.apply_leave(db, current_user, leave_request)


@router.get("/leaves/me/", response_model=list[LeaveSchema])
def get_my_leave_history(
    db: Session = Depends(get_db),
    current_user: Employee = Depends(get_current_user),
):
    """Return the currently authenticated employee's leave history."""
    return leave_service.get_my_leave_history(db, current_user)


@router.get("/leaves/pending/")
def get_pending_leaves(
    db: Session = Depends(get_db),
    manager: Employee = Depends(require_manager),
):
    """Manager-only pending requests with employee names.

    The internal database ID is returned to the Streamlit layer so it can
    safely call the approve/reject endpoint, but the UI does not need to
    display that internal ID to the manager.
    """
    pending = (
        db.query(LeaveModel)
        .filter(LeaveModel.status == "Pending")
        .order_by(LeaveModel.start_date.asc(), LeaveModel.id.asc())
        .all()
    )

    results = []
    for leave in pending:
        employee = (
            db.query(Employee)
            .filter(Employee.id == leave.employee_id)
            .first()
        )

        employee_name = "Unknown employee"
        if employee:
            employee_name = f"{employee.first_name} {employee.last_name}".strip()

        results.append({
            "id": leave.id,
            "employee_id": leave.employee_id,
            "employee_name": employee_name,
            "leave_type": leave.leave_type,
            "start_date": leave.start_date.isoformat(),
            "end_date": leave.end_date.isoformat(),
            "status": leave.status,
        })

    return results


@router.put("/leaves/{leave_id}/approve", response_model=LeaveSchema)
def approve_leave(
    leave_id: int,
    db: Session = Depends(get_db),
    manager: Employee = Depends(require_manager),
):
    """Manager-only approval."""
    return leave_service.approve_leave(db, leave_id)


@router.put("/leaves/{leave_id}/reject", response_model=LeaveSchema)
def reject_leave(
    leave_id: int,
    db: Session = Depends(get_db),
    manager: Employee = Depends(require_manager),
):
    """Manager-only rejection."""
    return leave_service.reject_leave(db, leave_id)


@router.put("/leaves/{leave_id}/cancel", response_model=LeaveSchema)
def cancel_leave(
    leave_id: int,
    db: Session = Depends(get_db),
    current_user: Employee = Depends(get_current_user),
):
    """Cancel only a leave belonging to the currently authenticated employee."""
    leave = db.query(LeaveModel).filter(LeaveModel.id == leave_id).first()

    if not leave:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Leave request not found")

    if leave.employee_id != current_user.id:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="You can only cancel your own leave requests")

    return leave_service.cancel_leave(db, leave_id)

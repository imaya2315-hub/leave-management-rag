from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_admin, require_manager
from app.models.employee import Employee
from app.models.leave import Leave as LeaveModel
from app.models.team import Team
from app.schemas.leave import Leave as LeaveSchema, LeaveCreate
from app.services import leave_service

router = APIRouter(tags=["Leaves"])


@router.post("/leaves/", response_model=LeaveSchema)
def apply_leave(
    leave_request: LeaveCreate,
    db: Session = Depends(get_db),
    current_user: Employee = Depends(get_current_user),
):
    return leave_service.apply_leave(db, current_user, leave_request)


@router.get("/leaves/me/", response_model=list[LeaveSchema])
def get_my_leave_history(
    db: Session = Depends(get_db),
    current_user: Employee = Depends(get_current_user),
):
    return leave_service.get_my_leave_history(db, current_user)


def _serialize_pending(leave: LeaveModel) -> dict:
    return {
        "id": leave.id,
        "employee_id": leave.employee_id,
        "employee_name": f"{leave.owner.first_name} {leave.owner.last_name}".strip(),
        "requester_role": leave.owner.role,
        "team_id": leave.owner.team_id,
        "leave_type": leave.leave_type,
        "start_date": leave.start_date.isoformat(),
        "end_date": leave.end_date.isoformat(),
        "status": leave.status,
    }


@router.get("/leaves/pending/")
def get_pending_leaves(
    db: Session = Depends(get_db),
    current_user: Employee = Depends(get_current_user),
):
    """Return only requests this user is allowed to review."""
    if current_user.role == "admin":
        pending = (
            db.query(LeaveModel)
            .join(Employee, Employee.id == LeaveModel.employee_id)
            .filter(
                LeaveModel.status == "Pending",
                Employee.role == "manager",
            )
            .order_by(LeaveModel.start_date.asc(), LeaveModel.id.asc())
            .all()
        )
        return [_serialize_pending(x) for x in pending]

    if current_user.role != "manager":
        raise HTTPException(status_code=403, detail="Only managers and admins can view pending leave requests")

    team = db.query(Team).filter(Team.manager_id == current_user.id).first()
    if not team:
        raise HTTPException(status_code=403, detail="You are not assigned to a team")

    pending = (
        db.query(LeaveModel)
        .join(Employee, Employee.id == LeaveModel.employee_id)
        .filter(
            LeaveModel.status == "Pending",
            Employee.team_id == team.id,
            Employee.role == "employee",
        )
        .order_by(LeaveModel.start_date.asc(), LeaveModel.id.asc())
        .all()
    )
    return [_serialize_pending(x) for x in pending]


def _authorize_approval(db: Session, approver: Employee, leave_id: int) -> LeaveModel:
    leave = db.query(LeaveModel).filter(LeaveModel.id == leave_id).first()
    if not leave:
        raise HTTPException(status_code=404, detail="Leave request not found")
    if leave.status != "Pending":
        raise HTTPException(status_code=400, detail=f"Leave is already {leave.status}, cannot review")
    if leave.employee_id == approver.id:
        raise HTTPException(status_code=403, detail="You cannot approve or reject your own leave request")

    requester = db.query(Employee).filter(Employee.id == leave.employee_id).first()
    if not requester:
        raise HTTPException(status_code=404, detail="Employee who owns this leave was not found")

    if requester.role == "admin":
        raise HTTPException(status_code=403, detail="Admin leave requests cannot be approved through the manager approval workflow")

    if requester.role == "manager":
        if approver.role != "admin":
            raise HTTPException(status_code=403, detail="Manager leave requests can only be approved or rejected by an admin")
        return leave

    # Employee request -> only the employee's team manager may approve/reject.
    if approver.role != "manager":
        raise HTTPException(status_code=403, detail="Employee leave requests can only be approved or rejected by a team manager")

    team = db.query(Team).filter(Team.manager_id == approver.id).first()
    if not team:
        raise HTTPException(status_code=403, detail="You are not assigned to a team")
    if requester.team_id != team.id:
        raise HTTPException(status_code=403, detail="You can only approve or reject leave requests from your team")
    return leave


@router.put("/leaves/{leave_id}/approve", response_model=LeaveSchema)
def approve_leave(
    leave_id: int,
    db: Session = Depends(get_db),
    approver: Employee = Depends(get_current_user),
):
    _authorize_approval(db, approver, leave_id)
    return leave_service.approve_leave(db, leave_id)


@router.put("/leaves/{leave_id}/reject", response_model=LeaveSchema)
def reject_leave(
    leave_id: int,
    db: Session = Depends(get_db),
    approver: Employee = Depends(get_current_user),
):
    _authorize_approval(db, approver, leave_id)
    return leave_service.reject_leave(db, leave_id)


@router.put("/leaves/{leave_id}/cancel", response_model=LeaveSchema)
def cancel_leave(
    leave_id: int,
    db: Session = Depends(get_db),
    current_user: Employee = Depends(get_current_user),
):
    leave = db.query(LeaveModel).filter(LeaveModel.id == leave_id).first()
    if not leave:
        raise HTTPException(status_code=404, detail="Leave request not found")
    if leave.employee_id != current_user.id:
        raise HTTPException(status_code=403, detail="You can only cancel your own leave requests")
    return leave_service.cancel_leave(db, leave_id)

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin, require_manager
from app.models.employee import Employee
from app.schemas.team import TeamCreate, TeamMemberSummary, TeamResponse
from app.services import team_service

router = APIRouter(prefix="/teams", tags=["Teams"])


@router.post("/", response_model=TeamResponse)
def create_team(
    team_request: TeamCreate,
    db: Session = Depends(get_db),
    admin: Employee = Depends(require_admin),
):
    team = team_service.create_team(db, admin, team_request.name, team_request.manager_id)
    return team_service.serialize_team(team)


@router.get("/", response_model=list[TeamResponse])
def list_teams(
    db: Session = Depends(get_db),
    admin: Employee = Depends(require_admin),
):
    return [team_service.serialize_team(team) for team in team_service.list_teams(db)]


@router.get("/me", response_model=TeamResponse)
def get_my_team(
    db: Session = Depends(get_db),
    manager: Employee = Depends(require_manager),
):
    team = team_service.get_my_team(db, manager)
    return team_service.serialize_team(team)


@router.put("/{team_id}/members/{employee_id}", response_model=TeamMemberSummary)
def add_member(
    team_id: int,
    employee_id: int,
    db: Session = Depends(get_db),
    admin: Employee = Depends(require_admin),
):
    return team_service.add_member(db, admin, team_id, employee_id)


@router.delete("/{team_id}/members/{employee_id}", response_model=TeamMemberSummary)
def remove_member(
    team_id: int,
    employee_id: int,
    db: Session = Depends(get_db),
    admin: Employee = Depends(require_admin),
):
    return team_service.remove_member(db, admin, team_id, employee_id)

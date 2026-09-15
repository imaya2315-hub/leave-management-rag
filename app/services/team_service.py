from fastapi import HTTPException
from sqlalchemy.orm import Session

from app import crud
from app.models.employee import Employee
from app.models.team import Team


def _get_team_or_404(db: Session, team_id: int) -> Team:
    team = crud.team.get_team(db, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return team


def create_team(db: Session, admin: Employee, name: str, manager_id: int) -> Team:
    if admin.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can create teams")
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Team name cannot be empty")
    if crud.team.get_team_by_name(db, name):
        raise HTTPException(status_code=400, detail="A team with this name already exists")

    manager = crud.employee.get_employee(db, manager_id)
    if not manager:
        raise HTTPException(status_code=404, detail="Manager not found")
    if manager.role != "manager":
        raise HTTPException(status_code=400, detail="The assigned employee must have manager role")
    if crud.team.get_team_by_manager(db, manager.id):
        raise HTTPException(status_code=400, detail="This manager already manages a team")
    if manager.team_id is not None:
        raise HTTPException(status_code=400, detail="This manager is already assigned to a team")

    team = crud.team.create_team(db, name, manager.id)
    manager.team_id = team.id
    db.commit()
    db.refresh(team)
    db.refresh(manager)
    return team


def add_member(db: Session, admin: Employee, team_id: int, employee_id: int) -> Employee:
    team = _get_team_or_404(db, team_id)
    employee = crud.employee.get_employee(db, employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    if employee.id == team.manager_id:
        employee.team_id = team.id
        db.commit()
        db.refresh(employee)
        return employee
    if employee.role == "manager":
        raise HTTPException(status_code=400, detail="A manager must be assigned through team creation and cannot join another team")
    if employee.team_id is not None and employee.team_id != team.id:
        raise HTTPException(status_code=400, detail="Employee already belongs to another team")
    employee.team_id = team.id
    db.commit()
    db.refresh(employee)
    return employee


def remove_member(db: Session, admin: Employee, team_id: int, employee_id: int) -> Employee:
    team = _get_team_or_404(db, team_id)
    employee = crud.employee.get_employee(db, employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    if employee.id == team.manager_id:
        raise HTTPException(status_code=400, detail="The team manager cannot be removed from the team")
    if employee.team_id != team_id:
        raise HTTPException(status_code=400, detail="Employee is not a member of this team")
    return crud.team.set_employee_team(db, employee, None)


def get_my_team(db: Session, manager: Employee) -> Team:
    if manager.role != "manager":
        raise HTTPException(status_code=403, detail="Only managers have a team view")
    team = crud.team.get_team_by_manager(db, manager.id)
    if not team:
        raise HTTPException(status_code=404, detail="You are not assigned to a team")
    return team


def list_teams(db: Session):
    return crud.team.get_teams(db)


def serialize_team(team: Team) -> dict:
    manager = team.manager
    return {
        "id": team.id,
        "name": team.name,
        "manager_id": team.manager_id,
        "manager_name": f"{manager.first_name} {manager.last_name}".strip(),
        "members": [
            {
                "id": member.id,
                "first_name": member.first_name,
                "last_name": member.last_name,
                "email": member.email,
                "role": member.role,
                "team_id": member.team_id,
            }
            for member in team.members
        ],
    }

from sqlalchemy.orm import Session

from app.models.employee import Employee
from app.models.team import Team


def get_team(db: Session, team_id: int) -> Team | None:
    return db.query(Team).filter(Team.id == team_id).first()


def get_team_by_name(db: Session, name: str) -> Team | None:
    return db.query(Team).filter(Team.name == name).first()


def get_team_by_manager(db: Session, manager_id: int) -> Team | None:
    return db.query(Team).filter(Team.manager_id == manager_id).first()


def create_team(db: Session, name: str, manager_id: int) -> Team:
    team = Team(name=name.strip(), manager_id=manager_id)
    db.add(team)
    db.commit()
    db.refresh(team)
    return team


def set_employee_team(db: Session, employee: Employee, team_id: int | None) -> Employee:
    employee.team_id = team_id
    db.commit()
    db.refresh(employee)
    return employee


def get_teams(db: Session) -> list[Team]:
    return db.query(Team).order_by(Team.id).all()

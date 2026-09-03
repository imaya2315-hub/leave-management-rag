from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_manager, require_service_client
from app.models.api_client import ApiClient
from app.models.employee import Employee as EmployeeModel
from app.schemas.employee import Employee, EmployeeCreate, EmployeeSummary
from app.services import employee_service

router = APIRouter(prefix="/employees", tags=["Employees"])


@router.post("/", response_model=Employee)
def create_employee(employee: EmployeeCreate, db: Session = Depends(get_db)):
    return employee_service.register_employee(db, employee)


@router.post("/service-register", response_model=Employee)
def create_employee_via_service(
    employee: EmployeeCreate,
    db: Session = Depends(get_db),
    client: ApiClient = Depends(require_service_client),
):
    """
    Same as POST /employees/, but reserved for trusted automation
    clients authenticated via OAuth2 Client Credentials (e.g. an n8n
    workflow) instead of the open public endpoint. Use this from n8n
    once it's configured with a client_id/client_secret, for a more
    locked-down registration path than the public endpoint above.
    """
    return employee_service.register_employee(db, employee)


@router.get("/", response_model=list[EmployeeSummary])
def get_employees(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    manager: EmployeeModel = Depends(require_manager),
):
    """Manager-only. Returns a lightweight employee list, no nested leave data."""
    return employee_service.list_employees(db, skip, limit)


@router.get("/{employee_id}", response_model=Employee)
def get_employee(employee_id: int, db: Session = Depends(get_db)):
    return employee_service.get_employee_or_404(db, employee_id)


@router.put("/{employee_id}", response_model=Employee)
def update_employee(employee_id: int, employee_update: EmployeeCreate, db: Session = Depends(get_db)):
    return employee_service.update_employee(db, employee_id, employee_update)


@router.delete("/{employee_id}")
def delete_employee(
    employee_id: int,
    db: Session = Depends(get_db),
    manager: EmployeeModel = Depends(require_manager),
):
    """Manager-only."""
    return employee_service.delete_employee(db, employee_id)

import jwt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.security import create_access_token, create_refresh_token, decode_token, verify_password
from app.crud.employee import get_employee_by_email
from app.models.employee import Employee
from app.schemas.employee import Employee as EmployeeSchema
from app.schemas.token import RefreshRequest, Token

router = APIRouter(tags=["Authentication"])


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    employee = get_employee_by_email(db, form_data.username)

    if not employee or not verify_password(form_data.password, employee.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = {"sub": employee.email, "id": employee.id}
    access_token = create_access_token(data=token_data)
    refresh_token = create_refresh_token(data=token_data)

    return Token(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=Token)
def refresh(payload: RefreshRequest, db: Session = Depends(get_db)):
    """
    Exchanges a valid, unexpired refresh token for a brand new access
    token (and a fresh refresh token) — without needing the user's
    email/password again. Call this when a request fails with 401
    because the access token expired.
    """
    invalid_refresh_exception = HTTPException(
        status_code=401,
        detail="Refresh token is invalid or expired — please log in again",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        decoded = decode_token(payload.refresh_token)
        email = decoded.get("sub")
        # Reject an access token being used here — only a genuine
        # refresh token should be able to mint new access tokens.
        if email is None or decoded.get("type") != "refresh":
            raise invalid_refresh_exception
    except jwt.PyJWTError:
        raise invalid_refresh_exception

    employee = get_employee_by_email(db, email)
    if employee is None:
        raise invalid_refresh_exception

    token_data = {"sub": employee.email, "id": employee.id}
    new_access_token = create_access_token(data=token_data)
    new_refresh_token = create_refresh_token(data=token_data)

    return Token(access_token=new_access_token, refresh_token=new_refresh_token)


@router.get("/employees/me/", response_model=EmployeeSchema)
def read_my_profile(current_user: Employee = Depends(get_current_user)):
    return current_user

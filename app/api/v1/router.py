from fastapi import APIRouter

from app.api.v1 import auth, employees, leaves, oauth, teams

api_router = APIRouter()

api_router.include_router(employees.router)
api_router.include_router(leaves.router)
api_router.include_router(auth.router)
api_router.include_router(oauth.router)
api_router.include_router(teams.router)

"""
Combines every v1 router into a single router, so main.py only has to
include one thing instead of several.
"""
from fastapi import APIRouter

from app.api.v1 import auth, assistant, employees, knowledge, leaves, oauth

api_router = APIRouter()

api_router.include_router(employees.router)
api_router.include_router(leaves.router)
api_router.include_router(auth.router)
api_router.include_router(oauth.router)
api_router.include_router(knowledge.router)
api_router.include_router(assistant.router)

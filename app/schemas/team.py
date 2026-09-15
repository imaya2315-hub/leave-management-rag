from pydantic import BaseModel, ConfigDict


class TeamCreate(BaseModel):
    name: str
    manager_id: int


class TeamMemberSummary(BaseModel):
    id: int
    first_name: str
    last_name: str
    email: str
    role: str
    team_id: int | None

    model_config = ConfigDict(from_attributes=True)


class TeamResponse(BaseModel):
    id: int
    name: str
    manager_id: int
    manager_name: str
    members: list[TeamMemberSummary]

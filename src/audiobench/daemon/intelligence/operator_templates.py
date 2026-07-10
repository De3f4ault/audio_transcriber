from typing import TypedDict, Any, ClassVar
from pydantic import BaseModel, Field

class DaemonProposal(TypedDict):
    region_id: str
    operator_template: str
    parameters: dict[str, Any]
    schema_version: int

class LectureModeOperator(BaseModel):
    SCHEMA_VERSION: ClassVar[int] = 2
    frequency_multiplier: float = Field(ge=0.1, le=10.0)

_TEMPLATE_CLASSES: dict[str, type[BaseModel]] = {
    "LectureModeOperator": LectureModeOperator,
}

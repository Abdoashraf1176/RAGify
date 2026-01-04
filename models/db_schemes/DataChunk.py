from pydantic import BaseModel, Field, validator
from typing import Optional, Dict
from bson import ObjectId


class DataChunk(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    chunk_text: str = Field(..., min_length=1)
    chunk_metadata: Dict
    chunk_order: int = Field(..., gt=0)
    chunk_project_id: str
    chunk_asset_id: str

    @classmethod
    def from_mongo(cls, data):
        if data:
            data["id"] = str(data.pop("_id"))
        return cls(**data)

    @classmethod
    def get_indexes(cls):
        return [
            {
                "key": [
                    ("chunk_project_id", 1)
                ],
                "name": "chunk_project_id_index_1",
                "unique": False
            }
        ]

    class Config:
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

class RetrievedDocument(BaseModel):
    text: str
    score: float
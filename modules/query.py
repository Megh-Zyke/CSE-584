from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    query: str
    response: str

class QueryRequest(BaseModel):
    query: str
    history: Optional[List[Message]] = None

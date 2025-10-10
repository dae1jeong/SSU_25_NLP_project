from pydantic import BaseModel, EmailStr

class EmailRequest(BaseModel):
    email: EmailStr

class VerifyRequest(BaseModel):
    email: EmailStr
    code: str

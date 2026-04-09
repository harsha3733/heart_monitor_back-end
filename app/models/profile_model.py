from pydantic import BaseModel, Field

class Profile(BaseModel):
    age: int
    bmi: float
    sex: int = Field(..., ge=0, le=1)  # 0=female, 1=male
    smoking: int = Field(..., ge=0, le=2)   # 0=no,1=occ,2=regular
    alcohol: int = Field(..., ge=0, le=2)   # 0=no,1=occ,2=regular
    diabetes: int = Field(..., ge=0, le=1)
    systolic_bp: int
    diastolic_bp: int
    cholesterol: int
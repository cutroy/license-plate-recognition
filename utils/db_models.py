from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import datetime
from typing import List, Optional
from pydantic import BaseModel

# модели базы данных
Base = declarative_base()

class DetectionSession(Base):
    __tablename__ = "detection_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    video_filename = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    output_filename = Column(String)
    frames_processed = Column(Integer, default=0)
    status = Column(String, default="pending")
    
    license_plates = relationship("LicensePlate", back_populates="session", cascade="all, delete-orphan")

class LicensePlate(Base):
    __tablename__ = "license_plates"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("detection_sessions.id"))
    plate_text = Column(String)
    confidence = Column(Float)
    frame_number = Column(Integer)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    vehicle_type = Column(String)
    detection_method = Column(String)
    
    session = relationship("DetectionSession", back_populates="license_plates")

# модели для API
class PlateSchema(BaseModel):
    id: Optional[int] = None
    text: str
    confidence: float
    frame_number: int
    timestamp: datetime.datetime
    vehicle_type: str
    method: str
    
    class Config:
        from_attributes = True

class SessionSchema(BaseModel):
    id: Optional[int] = None
    video_filename: str
    timestamp: datetime.datetime
    output_filename: Optional[str] = None
    frames_processed: int = 0
    status: str = "pending"
    license_plates: List[PlateSchema] = []
    
    class Config:
        from_attributes = True

class SessionCreate(BaseModel):
    video_filename: str

class SessionUpdate(BaseModel):
    frames_processed: Optional[int] = None
    status: Optional[str] = None
    output_filename: Optional[str] = None

# подключение к БД
def get_db_engine(db_url="sqlite:///license_plate_detection.db"):
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    return engine

def get_db_session(engine):
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return Session() 
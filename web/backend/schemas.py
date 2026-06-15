from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


class AllocateReq(BaseModel):
    files: List[str]


class UploadTarget(BaseModel):
    idx: int
    filename: str
    object_path: str
    url: str


class AllocateResp(BaseModel):
    job_id: str
    uploads: List[UploadTarget]


class SubmitReq(BaseModel):
    job_id: str


class VideoStatus(BaseModel):
    idx: int
    filename: str
    status: str
    duration_s: Optional[float] = None
    t_mix_90_s: Optional[float] = None
    t_mix_95_s: Optional[float] = None
    t_mix_99_s: Optional[float] = None
    error: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    status: str
    video_count: int
    videos: List[VideoStatus]

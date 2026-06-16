from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


class FileMeta(BaseModel):
    name: str
    size: int


class AllocateReq(BaseModel):
    files: List[FileMeta]


class UploadTarget(BaseModel):
    video_id: str
    filename: str
    object_path: str
    initiate_url: str   # signed resumable-initiate (POST) URL


class AllocateResp(BaseModel):
    uploads: List[UploadTarget]


class FinalizeReq(BaseModel):
    video_id: str
    filename: str
    object_path: str
    size_bytes: int


class VideoOut(BaseModel):
    video_id: str
    filename: str
    date: str
    size_bytes: int
    owner_email: str


class RunReq(BaseModel):
    video_ids: List[str]


class VideoStatus(BaseModel):
    idx: int
    video_id: str
    filename: str
    status: str
    duration_s: Optional[float] = None
    t_mix_90_s: Optional[float] = None
    t_mix_95_s: Optional[float] = None
    t_mix_99_s: Optional[float] = None
    error: Optional[str] = None


class RunStatus(BaseModel):
    run_id: str
    owner_email: str
    status: str
    video_count: int
    videos: List[VideoStatus]


class MeOut(BaseModel):
    email: str
    role: Optional[str]
    status: str


class SetUserReq(BaseModel):
    role: Optional[str] = None
    status: Optional[str] = None


class ManagedUser(BaseModel):
    email: str
    role: Optional[str]
    status: str
    decided_by: Optional[str] = None

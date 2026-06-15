from __future__ import annotations
from typing import Any, Dict

MAX_TASKS = 50  # cap fan-out to protect quota/cost


def build_overrides(job_id: str, bucket: str, video_count: int) -> Dict[str, Any]:
    if video_count < 1 or video_count > MAX_TASKS:
        raise ValueError(f"video_count {video_count} out of range 1..{MAX_TASKS}")
    return {"task_count": video_count, "container_overrides": [
        {"env": [{"name": "JOB_ID", "value": job_id}, {"name": "BUCKET", "value": bucket}]}]}


class JobRunner:
    def __init__(self, project: str, region: str, job_name: str):
        from google.cloud import run_v2
        self._client = run_v2.JobsClient()
        self._job_path = f"projects/{project}/locations/{region}/jobs/{job_name}"

    def trigger(self, job_id: str, bucket: str, video_count: int) -> str:
        from google.cloud import run_v2
        ov = build_overrides(job_id, bucket, video_count)
        overrides = run_v2.RunJobRequest.Overrides(
            task_count=ov["task_count"],
            container_overrides=[run_v2.RunJobRequest.Overrides.ContainerOverride(
                env=[run_v2.EnvVar(name=e["name"], value=e["value"])
                     for e in ov["container_overrides"][0]["env"]])])
        op = self._client.run_job(request=run_v2.RunJobRequest(name=self._job_path, overrides=overrides))
        return op.metadata.name if op.metadata else self._job_path

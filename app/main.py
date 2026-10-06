from __future__ import annotations

import math
import re
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .jobs import JobManager

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
FRAME_NAME = re.compile(r"frame_[0-9]{6}\.jpg")
manager = JobManager(BASE_DIR / "data")


class LocalJobRequest(BaseModel):
    path: str
    sample_fps: float = 2.0


@asynccontextmanager
async def lifespan(_: FastAPI):
    manager.start()
    try:
        yield
    finally:
        manager.stop()


app = FastAPI(title="DAV 猫影检测", lifespan=lifespan)


def _canonical_job_id(job_id: str) -> str:
    try:
        canonical = str(uuid.UUID(job_id))
    except (ValueError, AttributeError) as exc:
        raise HTTPException(status_code=404, detail="任务不存在") from exc
    if canonical != job_id.lower():
        raise HTTPException(status_code=404, detail="任务不存在")
    return canonical


def _ready_result(job_id: str) -> dict:
    try:
        return manager.load_ready_result(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="分析结果尚未就绪或任务不存在") from exc


@app.post("/api/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(...),
    sample_fps: float = Form(2.0),
) -> dict:
    if not math.isfinite(sample_fps) or not 0.5 <= sample_fps <= 5:
        raise HTTPException(status_code=422, detail="采样 FPS 必须在 0.5 到 5 之间")
    original_name = file.filename or ""
    if Path(original_name).suffix.lower() != ".dav":
        raise HTTPException(status_code=400, detail="请选择扩展名为 .dav 的监控视频")

    job_id = str(uuid.uuid4())
    job_dir = manager.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=False)
    source_path = job_dir / "source.dav"
    written = 0
    try:
        with source_path.open("xb") as output:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                output.write(chunk)
        if written == 0:
            raise HTTPException(status_code=400, detail="上传的视频为空")
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"保存上传文件失败：{exc}") from exc
    finally:
        await file.close()

    manager.create_job(job_id, sample_fps)
    manager.enqueue(job_id)
    return manager.get_job(job_id)


@app.post("/api/jobs/local", status_code=202)
def create_local_job(request: LocalJobRequest) -> dict:
    sample_fps = request.sample_fps
    if not math.isfinite(sample_fps) or not 0.5 <= sample_fps <= 5:
        raise HTTPException(status_code=422, detail="采样 FPS 必须在 0.5 到 5 之间")
    raw_path = request.path.strip()
    if len(raw_path) >= 2 and raw_path[0] == raw_path[-1] and raw_path[0] in {"'", '"'}:
        raw_path = raw_path[1:-1].strip()
    if not raw_path:
        raise HTTPException(status_code=400, detail="请输入 DAV 文件的绝对路径")
    source_path = Path(raw_path).expanduser()
    if not source_path.is_absolute():
        raise HTTPException(status_code=400, detail="请输入完整的绝对路径，例如 D:\\data\\video.dav")
    if source_path.suffix.lower() != ".dav":
        raise HTTPException(status_code=400, detail="文件扩展名必须是 .dav")
    try:
        source_path = source_path.resolve(strict=True)
        if not source_path.is_file():
            raise HTTPException(status_code=400, detail="所选路径不是文件")
        if source_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="所选文件为空")
    except HTTPException:
        raise
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=f"无法访问所选文件：{exc}") from exc

    job_id = str(uuid.uuid4())
    job_dir = manager.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=False)
    try:
        (job_dir / "source_path.txt").write_text(str(source_path), encoding="utf-8")
    except OSError as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"保存文件路径失败：{exc}") from exc
    manager.create_job(job_id, sample_fps, message="已连接本机文件，等待处理")
    manager.enqueue(job_id)
    return manager.get_job(job_id)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    canonical = _canonical_job_id(job_id)
    try:
        return manager.get_job(canonical)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在或结果已不可用") from exc


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str) -> FileResponse:
    canonical = _canonical_job_id(job_id)
    _ready_result(canonical)
    video_path = manager.job_dir(canonical) / "video.mp4"
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="MP4 视频文件不存在")
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/frames/{filename}")
def get_frame(job_id: str, filename: str) -> FileResponse:
    canonical = _canonical_job_id(job_id)
    if FRAME_NAME.fullmatch(filename) is None:
        raise HTTPException(status_code=404, detail="关键帧文件名无效")
    result = _ready_result(canonical)
    expected_url = f"/api/jobs/{canonical}/frames/{filename}"
    keyframes = result.get("keyframes", [])
    if not any(frame.get("imageUrl") == expected_url for frame in keyframes):
        raise HTTPException(status_code=404, detail="关键帧不属于该任务")
    frame_path = manager.job_dir(canonical) / "frames" / filename
    if not frame_path.is_file():
        raise HTTPException(status_code=404, detail="关键帧文件不存在")
    return FileResponse(frame_path, media_type="image/jpeg")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)

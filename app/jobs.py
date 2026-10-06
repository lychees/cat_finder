from __future__ import annotations

import copy
import json
import logging
import os
import queue
import threading
from pathlib import Path

from .analyzer import analyze_video

logger = logging.getLogger(__name__)


class JobManager:
    def __init__(self, data_dir: Path) -> None:
        self.jobs_dir = data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stopping = threading.Event()

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def start(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stopping.clear()
            self._worker = threading.Thread(
                target=self._run,
                name="dav-analysis-worker",
                daemon=True,
            )
            self._worker.start()

    def stop(self) -> None:
        self._stopping.set()
        self._queue.put(None)
        worker = self._worker
        if worker is not None and worker.is_alive() and worker != threading.current_thread():
            worker.join(timeout=5)

    def create_job(
        self,
        job_id: str,
        sample_fps: float,
        message: str = "上传完成，等待处理",
    ) -> None:
        with self._lock:
            self._jobs[job_id] = {
                "jobId": job_id,
                "status": "queued",
                "progress": 0.0,
                "message": message,
                "sampleFps": sample_fps,
            }

    def enqueue(self, job_id: str) -> None:
        if self._stopping.is_set():
            raise RuntimeError("服务正在停止，暂时不能接收新任务")
        self._queue.put(job_id)

    def get_job(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return copy.deepcopy(job)
        return self.load_ready_result(job_id)

    def load_ready_result(self, job_id: str) -> dict:
        result_path = self.job_dir(job_id) / "result.json"
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KeyError(job_id) from exc
        if (
            not isinstance(result, dict)
            or result.get("jobId") != job_id
            or result.get("status") != "ready"
        ):
            raise KeyError(job_id)
        return result

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(changes)

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                self._process(job_id)
            finally:
                self._queue.task_done()

    def _process(self, job_id: str) -> None:
        try:
            with self._lock:
                sample_fps = float(self._jobs[job_id]["sampleFps"])
            job_dir = self.job_dir(job_id)
            source_path = None
            source_pointer = job_dir / "source_path.txt"
            if source_pointer.is_file():
                source_path = Path(source_pointer.read_text(encoding="utf-8").strip())
                if not source_path.is_file():
                    raise RuntimeError("原始 DAV 文件已被移动、删除或暂时不可用")
            result_path = job_dir / "result.json"
            result_path.unlink(missing_ok=True)
            self._update(
                job_id,
                status="processing",
                progress=0.02,
                message="正在将 DAV 转码为 MP4",
            )

            def analysis_progress(fraction: float) -> None:
                fraction = min(1.0, max(0.0, fraction))
                self._update(
                    job_id,
                    progress=round(0.2 + fraction * 0.78, 4),
                    message=f"正在分析移动对象（{round(fraction * 100)}%）",
                )

            analysis = analyze_video(
                job_dir,
                sample_fps,
                analysis_progress,
                source_path=source_path,
            )
            job_id_prefix = f"/api/jobs/{job_id}"
            keyframes = []
            for keyframe in analysis["keyframes"]:
                public_keyframe = dict(keyframe)
                filename = public_keyframe.pop("filename")
                public_keyframe["imageUrl"] = f"{job_id_prefix}/frames/{filename}"
                keyframes.append(public_keyframe)
            result = {
                "jobId": job_id,
                "status": "ready",
                "progress": 1.0,
                "message": f"分析完成，共检测到 {len(keyframes)} 个移动关键帧",
                "sampleFps": sample_fps,
                "duration": analysis["duration"],
                "videoUrl": f"{job_id_prefix}/video",
                "keyframes": keyframes,
            }
            temporary = result_path.with_name("result.json.tmp")
            with temporary.open("w", encoding="utf-8") as result_file:
                json.dump(result, result_file, ensure_ascii=False, separators=(",", ":"))
            os.replace(temporary, result_path)
            with self._lock:
                self._jobs[job_id] = result
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            detail = str(exc).strip() or exc.__class__.__name__
            self._update(job_id, status="error", message=f"处理失败：{detail}")

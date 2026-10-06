from __future__ import annotations

import math
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .classifier import classify_crop

ANALYSIS_WIDTH = 640
PLAYBACK_WIDTH = 1280


def _transcode(source: Path, destination: Path) -> None:
    temporary = destination.with_name("video.part.mp4")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-fflags",
        "+genpts+discardcorrupt",
        "-err_detect",
        "ignore_err",
        "-threads",
        "4",
        "-i",
        str(source),
        "-an",
        "-vf",
        f"scale=trunc(min({PLAYBACK_WIDTH}\\,iw)/2)*2:-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-threads",
        "4",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            return_code = completed.returncode
            code_detail = (
                f"{return_code} / 0x{return_code & 0xFFFFFFFF:08X}"
                if return_code < 0
                else str(return_code)
            )
            lines = completed.stderr.strip().splitlines()
            detail = "\n".join(lines[-12:])[-2400:] or "FFmpeg 未输出诊断信息"
            raise RuntimeError(
                f"FFmpeg 转码失败（退出码 {code_detail}）：\n{detail}"
            )
        if not temporary.exists() or temporary.stat().st_size == 0:
            raise RuntimeError("FFmpeg 未生成有效的 MP4 文件")
        os.replace(temporary, destination)
    except FileNotFoundError as exc:
        raise RuntimeError("未找到 FFmpeg，请确认 ffmpeg 已加入 PATH") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _resize_for_analysis(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= ANALYSIS_WIDTH:
        return frame
    target_height = max(1, round(height * ANALYSIS_WIDTH / width))
    return cv2.resize(
        frame,
        (ANALYSIS_WIDTH, target_height),
        interpolation=cv2.INTER_AREA,
    )


def _motion_boxes(subtractor, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    frame_area = width * height
    mask = subtractor.apply(frame)
    mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=2,
    )
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    min_area = max(120.0, frame_area * 0.0005)
    min_side = max(10, round(width * 0.012))
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        box_area = box_width * box_height
        if box_width < min_side or box_height < min_side or box_area < min_area:
            continue
        if box_area >= frame_area * 0.85:
            continue
        padding = max(8, round(max(box_width, box_height) * 0.1))
        padded_x = max(0, x - padding)
        padded_y = max(0, y - padding)
        padded_right = min(width, x + box_width + padding)
        padded_bottom = min(height, y + box_height + padding)
        boxes.append(
            (
                padded_x,
                padded_y,
                padded_right - padded_x,
                padded_bottom - padded_y,
            )
        )
    gap = max(12, round(width * 0.02))
    merged = True
    while merged:
        merged = False
        for first_index, first in enumerate(boxes):
            first_x, first_y, first_width, first_height = first
            for second_index in range(first_index + 1, len(boxes)):
                second_x, second_y, second_width, second_height = boxes[second_index]
                if not (
                    first_x <= second_x + second_width + gap
                    and second_x <= first_x + first_width + gap
                    and first_y <= second_y + second_height + gap
                    and second_y <= first_y + first_height + gap
                ):
                    continue
                left = min(first_x, second_x)
                top = min(first_y, second_y)
                right = max(first_x + first_width, second_x + second_width)
                bottom = max(first_y + first_height, second_y + second_height)
                boxes[first_index] = (left, top, right - left, bottom - top)
                boxes.pop(second_index)
                merged = True
                break
            if merged:
                break
    boxes.sort(key=lambda box: (box[1], box[0]))
    return boxes


def analyze_video(
    job_dir: Path,
    sample_fps: float,
    progress_callback: Callable[[float], None],
    source_path: Path | None = None,
) -> dict:
    source = source_path if source_path is not None else job_dir / "source.dav"
    video_path = job_dir / "video.mp4"
    frames_dir = job_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    _transcode(source, video_path)
    progress_callback(0.0)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("OpenCV 无法打开转码后的 MP4")
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise RuntimeError("无法从转码后的视频中读取有效 FPS")
        frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=24,
            detectShadows=True,
        )
        keyframes: list[dict] = []
        sample_interval = 1.0 / sample_fps
        next_sample_time = 0.0
        frame_index = 0
        last_index = -1
        last_frame = None
        last_sampled_index = -1
        reported_progress = 0.0

        def report_progress(value: float) -> None:
            nonlocal reported_progress
            value = min(1.0, max(0.0, value))
            if value == 1.0 or value - reported_progress >= 0.01:
                progress_callback(value)
                reported_progress = value

        def process_sample(source_frame: np.ndarray, time_seconds: float) -> None:
            frame = _resize_for_analysis(source_frame)
            boxes = _motion_boxes(subtractor, frame)
            if not boxes:
                return
            height, width = frame.shape[:2]
            source_height, source_width = source_frame.shape[:2]
            scale_x = source_width / width
            scale_y = source_height / height
            detections: list[dict] = []
            for x, y, box_width, box_height in boxes:
                source_x = round(x * scale_x)
                source_y = round(y * scale_y)
                source_right = round((x + box_width) * scale_x)
                source_bottom = round((y + box_height) * scale_y)
                crop = source_frame[source_y:source_bottom, source_x:source_right]
                cat_confidence, top_label = classify_crop(crop)
                detections.append(
                    {
                        "x": round(x / width, 6),
                        "y": round(y / height, 6),
                        "w": round(box_width / width, 6),
                        "h": round(box_height / height, 6),
                        "catConfidence": round(cat_confidence, 6),
                        "topLabel": top_label,
                    }
                )
            annotated = frame.copy()
            for (x, y, box_width, box_height), detection in zip(boxes, detections):
                confidence = detection["catConfidence"]
                color = (50, 70, 235) if confidence >= 0.3 else (45, 180, 80)
                cv2.rectangle(
                    annotated,
                    (x, y),
                    (x + box_width, y + box_height),
                    color,
                    2,
                )
                short_label = detection["topLabel"].split(",")[0][:20]
                label = f"cat {confidence:.0%} | {short_label}"
                text_y = max(16, y - 7)
                cv2.putText(
                    annotated,
                    label,
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            filename = f"frame_{len(keyframes) + 1:06d}.jpg"
            success, encoded = cv2.imencode(
                ".jpg",
                annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            if not success:
                raise RuntimeError(f"关键帧 {filename} 编码失败")
            try:
                encoded.tofile(str(frames_dir / filename))
            except OSError as exc:
                raise RuntimeError(f"保存关键帧 {filename} 失败：{exc}") from exc
            keyframes.append(
                {
                    "time": round(time_seconds, 3),
                    "imageUrl": "",
                    "width": width,
                    "height": height,
                    "boxes": detections,
                    "filename": filename,
                }
            )

        while True:
            success, frame = capture.read()
            if not success:
                break
            time_seconds = frame_index / source_fps
            if frame_index == 0 or time_seconds + 1e-9 >= next_sample_time:
                process_sample(frame, time_seconds)
                last_sampled_index = frame_index
                while next_sample_time <= time_seconds + 1e-9:
                    next_sample_time += sample_interval
            last_frame = frame
            last_index = frame_index
            if frame_count > 0:
                report_progress(min(0.99, (frame_index + 1) / frame_count))
            frame_index += 1

        if last_index < 0 or last_frame is None:
            raise RuntimeError("转码后的视频没有可读取的画面")
        if last_sampled_index != last_index:
            process_sample(last_frame, last_index / source_fps)
        report_progress(1.0)
        duration = (last_index + 1) / source_fps
        return {
            "duration": round(duration, 3),
            "keyframes": keyframes,
        }
    finally:
        capture.release()

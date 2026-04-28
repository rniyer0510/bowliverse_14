from __future__ import annotations
from .shared import *

def _make_output_path(output_path: Optional[str]) -> str:
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return output_path
    name = f"skeleton_{uuid.uuid4().hex[:12]}.mp4"
    return os.path.join(RENDER_DIR, name)
def _intermediate_render_path(final_path: str) -> str:
    base_dir = os.path.dirname(final_path) or RENDER_DIR
    name = f"render_tmp_{uuid.uuid4().hex[:12]}.mp4"
    return os.path.join(base_dir, name)
def _publish_fallback_render(intermediate_path: str, final_path: str) -> str:
    try:
        os.replace(intermediate_path, final_path)
        return final_path
    except OSError:
        logger.warning(
            "[coach_video_renderer] Could not publish fallback render to final path; keeping intermediate encode",
        )
        return intermediate_path
def _finalize_render_video(intermediate_path: str, final_path: str) -> Tuple[str, str]:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        logger.warning(
            "[coach_video_renderer] ffmpeg not available; publishing mp4v fallback that may be unsupported on some browsers/iOS clients",
        )
        return _publish_fallback_render(intermediate_path, final_path), "mp4v"

    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        intermediate_path,
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        final_path,
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
    except Exception as exc:
        logger.warning("[coach_video_renderer] ffmpeg finalize failed: %s", exc)
        return _publish_fallback_render(intermediate_path, final_path), "mp4v"

    if completed.returncode != 0 or not os.path.exists(final_path):
        logger.warning(
            "[coach_video_renderer] ffmpeg finalize returned %s; publishing mp4v fallback that may be unsupported on some browsers/iOS clients",
            completed.returncode,
        )
        return _publish_fallback_render(intermediate_path, final_path), "mp4v"

    try:
        if os.path.exists(intermediate_path):
            os.remove(intermediate_path)
    except OSError:
        pass
    return final_path, "h264"

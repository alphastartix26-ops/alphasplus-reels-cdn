"""Infrastructure-only render + independent ASR probe. Never publishes to SNS.

Public inputs are a generic diagnostic script. No private source, account tokens,
client data, Actions artifact upload, or persistent Actions cache is used.
"""
from pathlib import Path
import difflib
import hashlib
import json
import math
import re
import subprocess
import sys


def run(args):
    return subprocess.run(args, check=True, capture_output=True)


def norm(text):
    return re.sub(r"[^가-힣A-Za-z0-9]", "", text).lower()


def main():
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    from faster_whisper import WhisperModel

    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("diagnostic") is not True:
        raise RuntimeError("THIS_PROBE_CANNOT_RELEASE_PRODUCTION_CONTENT")
    out = Path("cloud_render_output")
    out.mkdir(exist_ok=True)
    narration = manifest["narration"]
    voice = out / "voice.mp3"
    subtitles = out / "subtitles.srt"
    run([sys.executable, "-m", "edge_tts", "--voice", "ko-KR-HyunsuNeural", "--text", narration,
         "--write-media", str(voice), "--write-subtitles", str(subtitles)])
    if not voice.is_file() or voice.stat().st_size < 5000 or not subtitles.read_text().strip():
        raise RuntimeError("TTS_OR_ACTUAL_TIMED_CAPTIONS_MISSING")
    info = json.loads(run(["ffprobe", "-v", "quiet", "-show_format", "-of", "json", str(voice)]).stdout)
    duration = float(info["format"]["duration"])
    font_path = run(["fc-match", "-f", "%{file}", "Noto Sans CJK KR"]).stdout.decode()
    font = lambda n: ImageFont.truetype(font_path, n)
    cards = []
    headings = ["자동 제작 경로 점검", "한국어 음성 · 자막", "전체 영상 다시 분석", "운영 계정 게시 없음"]
    for number, text in enumerate(headings):
        image = Image.new("RGB", (1080, 1920), "#F6F5EE")
        draw = ImageDraw.Draw(image)
        draw.text((76, 100), "ALPHA S+ / INFRA TEST", font=font(40), fill="#8A6A35")
        draw.text((76, 520), text, font=font(59), fill="#182226")
        draw.text((76, 700), "진단용 영상 · 실제 뉴스 아님", font=font(40), fill="#626968")
        draw.rounded_rectangle((76, 930, 1004, 1120), radius=24, fill="#E8E2D2")
        draw.text((114, 980), "추가 유료 생성 API 사용 없음", font=font(42), fill="#725421")
        draw.text((76, 1540), "게시 자격: 없음", font=font(45), fill="#182226")
        path = out / f"card_{number}.png"
        image.save(path)
        cards.append(path)
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for card in cards:
        command += ["-loop", "1", "-framerate", "30", "-t", str(duration / len(cards)), "-i", str(card)]
    command += ["-i", str(voice)]
    graph = ";".join(f"[{i}:v]setsar=1,format=yuv420p[v{i}]" for i in range(len(cards)))
    graph += ";" + "".join(f"[v{i}]" for i in range(len(cards))) + "concat=n=4:v=1:a=0[base]"
    graph += ";[base]subtitles=" + str(subtitles) + ":force_style='Fontname=Noto Sans CJK KR,Fontsize=20,MarginV=160,Outline=2'[v]"
    video = out / "probe.mp4"
    command += ["-filter_complex", graph, "-map", "[v]", "-map", "4:a", "-af", "loudnorm=I=-18:TP=-1.5:LRA=9",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                "-r", "30", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(video)]
    run(command)
    sha = hashlib.sha256(video.read_bytes()).hexdigest()
    run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"])
    metadata = json.loads(run(["ffprobe", "-v", "quiet", "-show_streams", "-show_format", "-of", "json", str(video)]).stdout)
    wav = out / "final_audio.wav"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-ac", "1", "-ar", "16000", str(wav)])
    pcm = run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "s16le", "-ac", "1", "-ar", "16000", "-"]).stdout
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64) / 32768
    model = WhisperModel("medium", device="cpu", compute_type="int8", cpu_threads=4)
    segments, result_info = model.transcribe(str(wav), language="ko", initial_prompt=None,
                                           beam_size=5, word_timestamps=True, vad_filter=True)
    transcript = [{"start": segment.start, "end": segment.end, "text": segment.text,
                   "words": [{"start": word.start, "end": word.end, "word": word.word}
                             for word in (segment.words or [])]} for segment in segments]
    recognized = " ".join(segment["text"] for segment in transcript)
    similarity = difflib.SequenceMatcher(None, norm(narration), norm(recognized)).ratio()
    asr = {"video_sha256": sha, "model": "medium", "device": "cpu", "compute_type": "int8",
           "script_or_initial_prompt_supplied": False, "human_listening_performed": False,
           "duration_seconds": float(metadata["format"]["duration"]), "segments": transcript}
    (out / "asr.json").write_text(json.dumps(asr, ensure_ascii=False, indent=2))
    report = {"diagnostic": True, "publish_eligible": False, "operating_account_posted": False,
              "tts_created_on_cloud_runner": True, "timed_captions_created": True,
              "full_video_decode_passed": True, "video_sha256": sha,
              "video_size_bytes": video.stat().st_size, "duration_seconds": asr["duration_seconds"],
              "script_character_similarity": similarity, "asr_prompt_supplied": False,
              "asr_recognized_text": recognized, "human_listening_performed": False,
              "peak_dbfs": 20 * math.log10(max(abs(samples).max(), 1e-12)),
              "rms_dbfs": 20 * math.log10(max(np.sqrt((samples**2).mean()), 1e-12)),
              "clipped_sample_fraction": float((abs(samples) >= 1).mean()),
              "runner": "standard public-repository ubuntu-latest", "actions_cache_used": False,
              "actions_artifact_upload_used": False, "paid_generation_api_used": False,
              "metadata": metadata}
    (out / "BUILD_VERIFICATION.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if similarity < 0.93:
        raise RuntimeError("DIAGNOSTIC_ASR_MISMATCH")
    print(json.dumps({k: report[k] for k in ["video_sha256", "duration_seconds", "script_character_similarity", "publish_eligible"]}))


if __name__ == "__main__":
    main()

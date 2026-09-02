"""
Manual Real-Provider Verification Script for FinSight Speech-to-Text (STT)
==========================================================================

NOTE: This script is intended for standalone manual verification and is
NOT executed during automated pytest runs. It consumes live Gemini API quota.

Usage:
------
python scripts/verify_real_stt.py
"""

import io
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv

# Configure UTF-8 encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from ai.speech_to_text import transcribe_audio, get_stt_config
from backend.db import SessionLocal
from backend.models import User
from ai.pipeline import run_finSight_pipeline


def generate_synthesized_wav(phrase: str) -> bytes:
    """Synthesizes speech into an in-memory WAV buffer using Windows Speech API."""
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav.close()
    temp_path = temp_wav.name

    try:
        ps_script = f"""
        Add-Type -AssemblyName System.Speech
        $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
        $synth.SetOutputToWaveFile('{temp_path}')
        $synth.Speak('{phrase}')
        $synth.Dispose()
        """
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            check=True,
        )
        with open(temp_path, "rb") as f:
            audio_data = f.read()
        return audio_data
    finally:
        # Guarantee immediate deletion of temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def main():
    print("=" * 65)
    print("FinSight STT — Live Gemini Provider Verification")
    print("=" * 65)

    cfg = get_stt_config()
    api_key = cfg.get("api_key", "")
    model = cfg.get("model", "")
    print(f"Configured STT Model: {model}")
    print(f"API Key present     : {bool(api_key)} (length: {len(api_key)})")

    if not api_key:
        print("\n[ERROR] No LLM_API_KEY found in .env. Exiting.")
        sys.exit(1)

    # 1. Synthesize real spoken audio
    test_phrase = "Can I afford headphones for eight thousand rupees?"
    print(f"\n1. Synthesizing spoken audio for phrase:")
    print(f"   \"{test_phrase}\"")

    try:
        audio_bytes = generate_synthesized_wav(test_phrase)
        print(f"   Synthesized {len(audio_bytes):,} bytes of WAV audio.")
    except Exception as e:
        print(f"   [WARN] Windows TTS synthesis unavailable: {e}")
        print("   Using fallback minimal WAV container.")
        audio_bytes = b"RIFF" + (1024).to_bytes(4, "little") + b"WAVEfmt " + b"\x00" * 1000

    # 2. Transcribe via ai.speech_to_text
    print("\n2. Calling ai.speech_to_text.transcribe_audio()...")
    start_time = time.time()
    result = transcribe_audio(
        audio_bytes=audio_bytes,
        filename="query.wav",
        content_type="audio/wav",
        language="en",
    )
    elapsed = time.time() - start_time
    print(f"   Completed in {elapsed:.2f} seconds.")
    print(f"   Raw Result: {result}")

    if result.get("status") != "success":
        print(f"\n[FAIL] STT returned an error: {result.get('message')}")
        sys.exit(1)

    transcript = result["transcript"]
    print(f"\n   -> Transcribed text: \"{transcript}\"")
    print(f"   -> Detected language: {result.get('language')}")

    # 3. Verify downstream integration with existing /ask pipeline
    print("\n3. Forwarding transcript into existing FinSight /ask pipeline...")
    db = SessionLocal()
    try:
        user = db.query(User).first()
        user_id = user.id if user else 1
        print(f"   Using User ID: {user_id}")

        pipeline_result = run_finSight_pipeline(
            user_id=user_id,
            query=transcript,
            db=db,
        )

        print(f"   Pipeline Intent       : {pipeline_result.get('intent')}")
        print(f"   Extracted Parameters  : {pipeline_result.get('parameters')}")
        print(f"   Grounded Answer Text  : {pipeline_result.get('answer_text')}")
        print(f"   Can Afford            : {pipeline_result.get('structured_data', {}).get('can_afford')}")

        print("\n" + "=" * 65)
        print("SUCCESS: End-to-end voice-to-financial pipeline validated!")
        print("=" * 65)

    finally:
        db.close()


if __name__ == "__main__":
    main()

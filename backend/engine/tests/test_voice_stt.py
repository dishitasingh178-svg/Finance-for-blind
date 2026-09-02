"""
Integration & Unit Tests for FinSight Voice Speech-to-Text (STT) Layer.

Tests:
1. Successful audio transcription (200 OK, VoiceTranscribeResponse)
2. Empty audio rejection (400 Bad Request)
3. Unsupported media type rejection (415 Unsupported Media Type)
4. Provider timeout handling (504 Gateway Timeout)
5. Gemini rate limit 429 handling (429 Too Many Requests)
6. Gemini provider overload 503 handling (503 Service Unavailable)
7. Gemini authentication failure 401/403 handling (502 Bad Gateway)
8. Malformed Gemini response / safety block handling (400 Bad Request)
9. Hinglish transcript preservation without translation
10. API response schema validation via Pydantic models
11. Versioned endpoint parity (/voice/transcribe and /api/v1/voice/transcribe)
12. CRITICAL ARCHITECTURAL ISOLATION:
    Assert that during /voice/transcribe, ALL financial_engine.py methods have call_count == 0
    and neither ai.speech_to_text nor backend.routers.voice import financial_engine.
"""

import io
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
import httpx

from backend.main import app
from backend.schemas import VoiceTranscribeResponse, VoiceTranscribeErrorResponse
import backend.engine.financial_engine as real_financial_engine
import ai.speech_to_text
import backend.routers.voice


# Sample valid audio bytes (> 32 bytes with RIFF/WAV header pattern)
def make_dummy_audio_bytes(size: int = 128) -> bytes:
    """Creates dummy audio bytes with minimal WAV container header."""
    header = b"RIFF" + (size - 8).to_bytes(4, "little") + b"WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data" + (size - 44).to_bytes(4, "little")
    payload = b"\x00" * max(0, size - len(header))
    return header + payload


def mock_gemini_success_response(transcript_text: str) -> httpx.Response:
    """Builds a mock successful Gemini generateContent HTTP response."""
    return httpx.Response(
        status_code=200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": transcript_text}
                        ]
                    }
                }
            ]
        },
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com"),
    )


@pytest.fixture
def client():
    """TestClient fixture for FastAPI."""
    return TestClient(app)


class TestVoiceSpeechToText:
    """Test suite for Speech-to-Text service and endpoints."""

    def test_successful_transcription_webm(self, client: TestClient):
        """Test 1: Successful webm upload returns 200 and transcript."""
        expected_transcript = "What is my balance?"
        dummy_audio = make_dummy_audio_bytes(256)

        with patch.object(httpx.Client, "post", return_value=mock_gemini_success_response(expected_transcript)):
            response = client.post(
                "/voice/transcribe",
                files={"audio": ("recording.webm", dummy_audio, "audio/webm")},
                data={"language": "en"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["transcript"] == expected_transcript
        assert data["language"] == "en"
        # Validate Pydantic schema
        schema_obj = VoiceTranscribeResponse.model_validate(data)
        assert schema_obj.transcript == expected_transcript

    def test_successful_transcription_wav(self, client: TestClient):
        """Test 1b: Successful wav upload returns 200 and transcript."""
        expected_transcript = "Can I afford headphones for eight thousand?"
        dummy_audio = make_dummy_audio_bytes(256)

        with patch.object(httpx.Client, "post", return_value=mock_gemini_success_response(expected_transcript)):
            response = client.post(
                "/voice/transcribe",
                files={"audio": ("speech.wav", dummy_audio, "audio/wav")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["transcript"] == expected_transcript

    def test_empty_audio_rejection(self, client: TestClient):
        """Test 2: Empty audio file returns 400."""
        response = client.post(
            "/voice/transcribe",
            files={"audio": ("empty.wav", b"", "audio/wav")},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"
        assert "empty" in data["message"].lower() or "empty" in data.get("detail", "").lower()
        VoiceTranscribeErrorResponse.model_validate(data)

    def test_too_short_audio_rejection(self, client: TestClient):
        """Test 2b: Audio file < 32 bytes returns 400."""
        response = client.post(
            "/voice/transcribe",
            files={"audio": ("tiny.wav", b"RIFF123", "audio/wav")},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"
        assert "too short" in data["message"].lower() or "corrupt" in data["message"].lower()

    def test_unsupported_media_type_rejection(self, client: TestClient):
        """Test 3: Unsupported file extension returns 415."""
        response = client.post(
            "/voice/transcribe",
            files={"audio": ("notes.txt", b"Some text content that is not audio at all", "text/plain")},
        )

        assert response.status_code == 415
        data = response.json()
        assert data["status"] == "error"
        assert "unsupported" in data["message"].lower() or "unsupported" in data.get("detail", "").lower()
        VoiceTranscribeErrorResponse.model_validate(data)

    def test_provider_timeout_handling(self, client: TestClient):
        """Test 4: Provider timeout returns 504."""
        dummy_audio = make_dummy_audio_bytes(128)

        with patch.object(httpx.Client, "post", side_effect=httpx.TimeoutException("Read timed out")):
            response = client.post(
                "/voice/transcribe",
                files={"audio": ("recording.mp3", dummy_audio, "audio/mp3")},
            )

        assert response.status_code == 504
        data = response.json()
        assert data["status"] == "error"
        assert "timed out" in data["message"].lower()
        VoiceTranscribeErrorResponse.model_validate(data)

    def test_gemini_429_rate_limit_handling(self, client: TestClient):
        """Test 5: Gemini 429 quota exhaustion returns 429."""
        dummy_audio = make_dummy_audio_bytes(128)
        err_resp = httpx.Response(
            status_code=429,
            json={"error": {"code": 429, "message": "Resource has been exhausted (e.g. check quota)."}},
            request=httpx.Request("POST", "https://generativelanguage.googleapis.com"),
        )

        with patch.object(httpx.Client, "post", return_value=err_resp):
            response = client.post(
                "/voice/transcribe",
                files={"audio": ("recording.m4a", dummy_audio, "audio/mp4")},
            )

        assert response.status_code == 429
        data = response.json()
        assert data["status"] == "error"
        assert "rate limit" in data["message"].lower()
        VoiceTranscribeErrorResponse.model_validate(data)

    def test_gemini_503_service_unavailable_handling(self, client: TestClient):
        """Test 6: Gemini 503 high demand returns 503."""
        dummy_audio = make_dummy_audio_bytes(128)
        err_resp = httpx.Response(
            status_code=503,
            json={"error": {"code": 503, "message": "Model is experiencing high demand."}},
            request=httpx.Request("POST", "https://generativelanguage.googleapis.com"),
        )

        with patch.object(httpx.Client, "post", return_value=err_resp):
            response = client.post(
                "/voice/transcribe",
                files={"audio": ("recording.ogg", dummy_audio, "audio/ogg")},
            )

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "error"
        assert "demand" in data["message"].lower() or "unavailable" in data["message"].lower()
        VoiceTranscribeErrorResponse.model_validate(data)

    def test_gemini_authentication_failure_handling(self, client: TestClient):
        """Test 7: Gemini 401/403 auth error returns 502."""
        dummy_audio = make_dummy_audio_bytes(128)
        err_resp = httpx.Response(
            status_code=401,
            json={"error": {"code": 401, "message": "API key not valid. Please pass a valid API key."}},
            request=httpx.Request("POST", "https://generativelanguage.googleapis.com"),
        )

        with patch.object(httpx.Client, "post", return_value=err_resp):
            response = client.post(
                "/voice/transcribe",
                files={"audio": ("recording.wav", dummy_audio, "audio/wav")},
            )

        assert response.status_code == 502
        data = response.json()
        assert data["status"] == "error"
        assert "authentication" in data["message"].lower()
        VoiceTranscribeErrorResponse.model_validate(data)

    def test_malformed_gemini_response_handling(self, client: TestClient):
        """Test 8: Malformed Gemini JSON (empty candidates / safety block) returns 400."""
        dummy_audio = make_dummy_audio_bytes(128)
        empty_resp = httpx.Response(
            status_code=200,
            json={"candidates": []},
            request=httpx.Request("POST", "https://generativelanguage.googleapis.com"),
        )

        with patch.object(httpx.Client, "post", return_value=empty_resp):
            response = client.post(
                "/voice/transcribe",
                files={"audio": ("recording.wav", dummy_audio, "audio/wav")},
            )

        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"
        assert "couldn't detect" in data["message"].lower()
        VoiceTranscribeErrorResponse.model_validate(data)

    def test_silence_noise_token_handling(self, client: TestClient):
        """Test 8b: [NO_SPEECH] or <noise> token from model returns 400 error."""
        dummy_audio = make_dummy_audio_bytes(128)
        noise_resp = httpx.Response(
            status_code=200,
            json={"candidates": [{"content": {"parts": [{"text": "<noise>"}]}}]},
            request=httpx.Request("POST", "https://generativelanguage.googleapis.com"),
        )

        with patch.object(httpx.Client, "post", return_value=noise_resp):
            response = client.post(
                "/voice/transcribe",
                files={"audio": ("silent.wav", dummy_audio, "audio/wav")},
            )

        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"
        assert "couldn't detect" in data["message"].lower()

    def test_hinglish_transcript_preservation(self, client: TestClient):
        """Test 9: Verbatim Hinglish transcript is preserved without English translation."""
        hinglish_query = "Bhai mere account mein kitne paise hain?"
        dummy_audio = make_dummy_audio_bytes(256)

        with patch.object(httpx.Client, "post", return_value=mock_gemini_success_response(hinglish_query)):
            response = client.post(
                "/voice/transcribe",
                files={"audio": ("hinglish.webm", dummy_audio, "audio/webm")},
                data={"language": "hi"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        # Verbatim Hinglish preserved
        assert data["transcript"] == "Bhai mere account mein kitne paise hain?"
        assert "how much" not in data["transcript"].lower()  # NOT translated to English
        assert data["language"] == "hi"

    def test_endpoint_aliases_parity(self, client: TestClient):
        """Test 11: /voice/transcribe and /api/v1/voice/transcribe return identical structures."""
        transcript = "Show my recent transactions"
        dummy_audio = make_dummy_audio_bytes(256)

        with patch.object(httpx.Client, "post", return_value=mock_gemini_success_response(transcript)):
            resp1 = client.post(
                "/voice/transcribe",
                files={"audio": ("test.wav", dummy_audio, "audio/wav")},
            )
            resp2 = client.post(
                "/api/v1/voice/transcribe",
                files={"audio": ("test.wav", dummy_audio, "audio/wav")},
            )

        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json() == resp2.json()

    def test_architectural_isolation_financial_engine_never_called(self, client: TestClient):
        """
        Test 12 (CRITICAL ARCHITECTURAL TEST):
        Mock all financial_engine.py functions and verify that during /voice/transcribe:
        1. ALL financial engine methods have call_count == 0.
        2. ai.speech_to_text does not import financial_engine.
        3. backend.routers.voice does not import financial_engine.
        """
        # 1. Verify module imports do NOT contain financial engine
        stt_modules = set(dir(ai.speech_to_text))
        voice_router_modules = set(dir(backend.routers.voice))
        assert "financial_engine" not in stt_modules
        assert "financial_engine" not in voice_router_modules
        assert "real_engine" not in stt_modules
        assert "real_engine" not in voice_router_modules

        # 2. Spy on all key financial engine methods
        with patch.object(real_financial_engine, "get_balance") as mock_bal, \
             patch.object(real_financial_engine, "get_spending_summary") as mock_spend, \
             patch.object(real_financial_engine, "check_affordability") as mock_afford, \
             patch.object(real_financial_engine, "project_goal_completion") as mock_goal, \
             patch.object(real_financial_engine, "get_insights") as mock_insight, \
             patch.object(httpx.Client, "post", return_value=mock_gemini_success_response("Can I buy 8k shoes?")):

            response = client.post(
                "/voice/transcribe",
                files={"audio": ("speech.wav", make_dummy_audio_bytes(128), "audio/wav")},
            )

            assert response.status_code == 200
            assert response.json()["status"] == "success"

            # 3. Assert zero financial engine invocations
            assert mock_bal.call_count == 0, "financial_engine.get_balance was called by /voice/transcribe!"
            assert mock_spend.call_count == 0, "financial_engine.get_spending_summary was called by /voice/transcribe!"
            assert mock_afford.call_count == 0, "financial_engine.check_affordability was called by /voice/transcribe!"
            assert mock_goal.call_count == 0, "financial_engine.project_goal_completion was called by /voice/transcribe!"
            assert mock_insight.call_count == 0, "financial_engine.get_insights was called by /voice/transcribe!"

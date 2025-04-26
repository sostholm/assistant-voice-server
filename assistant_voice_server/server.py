# server.py

import asyncio
import websockets
import logging
import os
import collections
import webrtcvad
import tempfile
import wave
from elevenlabs import VoiceSettings, AsyncElevenLabs, Voice
from .database import get_users_voice_recognition, get_device_by_id, Device, DSN
import psycopg
from typing import AsyncIterator, List
import json
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

STT_SERVER_URI = os.getenv("STT_SERVER_URI", "ws://localhost:9000/ws")
AI_AGENT_URI = os.getenv("AI_AGENT_URI", "ws://localhost:8000/ws")

# ElevenLabs API configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")  # Ensure you set this environment variable
VOICE_ID = os.getenv("VOICE_ID")  # Ensure you set this environment variable

# Initialize ElevenLabs async client
client = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)

# VAD configuration
VAD_MODE = 3  # Most aggressive mode
vad = webrtcvad.Vad(VAD_MODE)

# Audio configuration
CHANNELS = 1
MIC_RATE = 16000  # Should match the client's MIC_RATE
CHUNK_DURATION_MS = 20  # Must match the client's CHUNK_DURATION_MS
CHUNK_SIZE = int(MIC_RATE * CHUNK_DURATION_MS / 1000) * 2  # 2 bytes per sample

# --- Triggering Configuration ---
SPEECH_TRIGGER_DURATION_MS = 200  # Pre-roll buffering / initial speech detection window
SPEECH_TRIGGER_FRAMES = int(SPEECH_TRIGGER_DURATION_MS / CHUNK_DURATION_MS)
SPEECH_TRIGGER_THRESHOLD = 0.8  # Min ratio of voiced frames in trigger window to start recording

# --- Silence Detection Configuration ---
SILENCE_DETECT_DURATION_MS = 700  # Duration of silence needed to end recording
SILENCE_DETECT_FRAMES = int(SILENCE_DETECT_DURATION_MS / CHUNK_DURATION_MS)
SILENCE_DETECT_THRESHOLD = 0.9  # Min ratio of non-voiced frames in silence window to stop recording

# --- Segment Configuration ---
MAX_SEGMENT_DURATION_S = 5  # Max duration before sending segment to STT
MAX_SEGMENT_FRAMES = int(MAX_SEGMENT_DURATION_S * 1000 / CHUNK_DURATION_MS)

AUTHORIZED_USERS: List[str] = [u.nick_name for u in get_users_voice_recognition() if u.human]

async def connect_to_websocket(uri):
    while True:
        try:
            return await websockets.connect(uri)
        except (websockets.ConnectionClosedError, ConnectionRefusedError):
            logger.warning(f"Failed to connect to WebSocket at {uri}. Retrying in 5 seconds...")
            await asyncio.sleep(5)

async def generate_tts_stream(text: str) -> AsyncIterator[bytes]:
    # Use the AsyncElevenLabs client to generate TTS audio stream
    voice = Voice(voice_id=VOICE_ID)
    voice_settings = VoiceSettings(
        stability=0.15,
        similarity_boost=0.75,
        style=0.75,
        use_speaker_boost=True,
    )

    stream = await client.generate(
        text=text,
        voice=voice,
        voice_settings=voice_settings,
        model="eleven_turbo_v2",
        stream=True,
        optimize_streaming_latency=0,  # Adjust as needed
        output_format="mp3_44100_128",
    )
    return stream

async def stream_tts_to_client(websocket, text: str):
    """Generates TTS audio and streams it to the client."""
    try:
        logger.info(f"Generating TTS for: {text[:50]}...") # Log beginning of TTS generation
        tts_stream = await generate_tts_stream(text)
        logger.info("Streaming TTS audio to client")
        async for chunk in tts_stream:
            if chunk:
                await websocket.send(chunk)
                await asyncio.sleep(0) # Yield control to allow other tasks (like keepalive) to run
        logger.info("Completed streaming TTS audio to client")
    except websockets.ConnectionClosed:
        logger.warning("Client connection closed during TTS streaming.")
    except Exception as e:
        logger.error(f"Error during TTS generation or streaming: {e}")

@dataclass
class AiAgentMessage:
    nickname: str
    message: str
    location: str

def filter_authorized_users(transcriptions: List[dict], device: Device) -> List[AiAgentMessage]:
    filtered_text = []
    # Check the line starts with <nickname>:
    for item in transcriptions:
        for key, value in item.items():
            if key in AUTHORIZED_USERS:
                filtered_text.append(AiAgentMessage(nickname=key, message=value, location=device.location))
    
    return filtered_text

async def handle_client(websocket):
    logger.info(f"Client connected: {websocket.remote_address}")

    try:
        device_id_msg = await websocket.recv()
        device_id = int(device_id_msg)

        async with await psycopg.AsyncConnection.connect(DSN) as conn:
            device: Device = await get_device_by_id(conn, device_id)
        assert device is not None, f"Device with ID {device_id} not found"
        
        logger.info(f"Received device ID from client: {device_id}")
    except ValueError:
        logger.error("Failed to parse device ID from client. Closing connection.")
        await websocket.close()
        return
    except websockets.ConnectionClosed:
        logger.info("Connection closed before receiving device ID.")
        return

    # Separate buffers for triggering and silence detection
    trigger_buffer = collections.deque(maxlen=SPEECH_TRIGGER_FRAMES)
    silence_buffer = collections.deque(maxlen=SILENCE_DETECT_FRAMES)
    
    triggered = False
    voiced_frames = []
    current_transcription_batch: List[AiAgentMessage] = []

    # WebSocket connections to STT and AI agent
    stt_uri = STT_SERVER_URI
    ai_agent_uri = AI_AGENT_URI

    stt_socket = await connect_to_websocket(stt_uri)
    ai_agent_socket = await connect_to_websocket(ai_agent_uri)

    # Send device information to AI Agent
    await ai_agent_socket.send(device_id_msg)

    async def listen_to_ai_agent():
        try:
            async for message in ai_agent_socket:
                logger.info(f"Received response from AI Agent: {message}")
                # Generate and stream TTS audio back to the client in a separate task
                # to avoid blocking the listener loop.
                asyncio.create_task(stream_tts_to_client(websocket, message))
        except websockets.ConnectionClosed:
            logger.warning("Connection to AI Agent closed.")
        except Exception as e:
            logger.error(f"Error listening to AI Agent: {e}")

    ai_listener_task = asyncio.create_task(listen_to_ai_agent())

    try:
        while True:
            message = await websocket.recv()
            if isinstance(message, bytes):
                chunk = message
                try:
                    is_speech = vad.is_speech(chunk, MIC_RATE)
                except webrtcvad.VadError as e:
                    logger.warning(f"VAD error processing chunk: {e}. Skipping chunk.")
                    continue

                if not triggered:
                    trigger_buffer.append((chunk, is_speech))
                    # Check if buffer is full enough to make a decision
                    if len(trigger_buffer) == trigger_buffer.maxlen:
                        num_voiced = len([f for f, speech in trigger_buffer if speech])
                        if num_voiced > SPEECH_TRIGGER_THRESHOLD * trigger_buffer.maxlen:
                            triggered = True
                            logger.info("Speech detected, start recording")
                            # Include pre-roll frames from trigger buffer
                            voiced_frames.extend([f for f, s in trigger_buffer])
                            trigger_buffer.clear() # Clear trigger buffer
                            silence_buffer.clear() # Ensure silence buffer is clear
                            current_transcription_batch = [] # Reset batch
                else:
                    # Append audio chunk to main list
                    voiced_frames.append(chunk)
                    # Append speech status to silence detection buffer
                    silence_buffer.append(is_speech)

                    # Check silence only if silence buffer is full
                    silence_detected = False
                    if len(silence_buffer) == silence_buffer.maxlen:
                        num_unvoiced = len([speech for speech in silence_buffer if not speech])
                        if num_unvoiced > SILENCE_DETECT_THRESHOLD * silence_buffer.maxlen:
                            silence_detected = True
                    
                    max_duration_reached = len(voiced_frames) >= MAX_SEGMENT_FRAMES

                    if silence_detected or max_duration_reached:
                        process_reason = "Silence" if silence_detected else "Max duration"
                        logger.info(f"{process_reason} detected, processing audio segment.")

                        frames_to_process = voiced_frames[:]
                        voiced_frames.clear()

                        if not frames_to_process:
                            logger.warning("Skipping processing of empty audio segment.")
                            if silence_detected:
                                logger.info("Resetting VAD state due to silence (empty segment).")
                                trigger_buffer.clear() # Clear both buffers on reset
                                silence_buffer.clear()
                                triggered = False
                                current_transcription_batch = []
                            continue

                        audio_data = b''.join(frames_to_process)
                        audio_file_path = None

                        try:
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                                with wave.open(temp_audio_file, "wb") as wf:
                                    wf.setnchannels(CHANNELS)
                                    wf.setsampwidth(2)  # 16-bit PCM
                                    wf.setframerate(MIC_RATE)
                                    wf.writeframes(audio_data)
                                audio_file_path = temp_audio_file.name

                            logger.info(f"Sending {len(audio_data)} bytes of audio data to STT service ({process_reason} trigger)")
                            with open(audio_file_path, "rb") as f:
                                await stt_socket.send(f.read())
                            transcription_json = await stt_socket.recv()
                            transcription = json.loads(transcription_json)
                            logger.info(f"Received transcription: {transcription}")

                            filtered_messages: List[AiAgentMessage] = filter_authorized_users(transcription, device=device)
                            authorized_user_in_segment = len(filtered_messages) > 0
                            logger.info(f"Authorized user in segment: {authorized_user_in_segment}")

                            current_transcription_batch.extend(filtered_messages)
                            batch_contains_authorized = any(msg.nickname in AUTHORIZED_USERS for msg in current_transcription_batch)

                            send_batch_to_ai = False
                            reset_vad_state = False

                            if silence_detected:
                                if batch_contains_authorized:
                                    logger.info("Silence detected and batch contains authorized user. Sending batch to AI.")
                                    send_batch_to_ai = True
                                else:
                                    logger.info("Silence detected but batch has no authorized user. Discarding batch.")
                                reset_vad_state = True # Always reset on silence
                            elif max_duration_reached:
                                if not authorized_user_in_segment:
                                    if batch_contains_authorized:
                                        logger.info("Max duration reached, last segment had no authorized user. Sending accumulated batch to AI.")
                                        send_batch_to_ai = True
                                    else:
                                        logger.info("Max duration reached, no authorized user in last segment or batch. Discarding batch.")
                                    reset_vad_state = True # Reset if authorized speech stream broken
                                else:
                                    logger.info("Max duration reached with authorized user. Continuing batch.")
                                    send_batch_to_ai = False
                                    reset_vad_state = False # Continue listening

                            if send_batch_to_ai and current_transcription_batch:
                                json_data = json.dumps([asdict(message) for message in current_transcription_batch])
                                logger.info(f"Sending batch to AI agent: {json_data}")
                                await ai_agent_socket.send(json_data)

                            if reset_vad_state:
                                logger.info("Resetting VAD state.")
                                trigger_buffer.clear() # Clear both buffers
                                silence_buffer.clear()
                                triggered = False
                                current_transcription_batch = []

                        except websockets.ConnectionClosed as e:
                            logger.warning(f"Connection to STT or AI Agent closed during processing: {e}")
                            if silence_detected: # Check original trigger reason for reset
                                logger.info("Resetting VAD state due to silence after connection error.")
                                trigger_buffer.clear()
                                silence_buffer.clear()
                                triggered = False
                                current_transcription_batch = []
                            continue
                        except Exception as e:
                            logger.error(f"Error during STT/AI processing: {e}", exc_info=True)
                            if silence_detected: # Check original trigger reason for reset
                                logger.info("Resetting VAD state due to silence after processing error.")
                                trigger_buffer.clear()
                                silence_buffer.clear()
                                triggered = False
                                current_transcription_batch = []
                        finally:
                            if audio_file_path and os.path.exists(audio_file_path):
                                os.remove(audio_file_path)

            else:
                logger.warning("Received non-bytes message from client")
    except websockets.ConnectionClosed:
        logger.info(f"Connection closed: {websocket.remote_address}")
    except Exception as e:
        logger.error(f"Error handling client {websocket.remote_address}: {e}", exc_info=True)
    finally:
        logger.info(f"Client disconnected: {websocket.remote_address}")
        await stt_socket.close()
        await ai_agent_socket.close()
        ai_listener_task.cancel()

async def main():
    # Add ping_interval and ping_timeout to keep connections alive
    server = await websockets.serve(
        handle_client,
        '0.0.0.0',
        9001,
        ping_interval=20,  # Send a ping every 20 seconds
        ping_timeout=20   # Close connection if pong not received within 20 seconds
    )
    logger.info("Server started on port 9001 with keepalive pings enabled")
    await server.wait_closed()

if __name__ == "__main__":

    asyncio.run(main())

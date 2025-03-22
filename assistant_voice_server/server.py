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
SILENCE_THRESHOLD = 0.5  # seconds of silence to consider as end of speech
VOICED_THRESHOLD = 0.8  # Threshold for speech detection
PRE_ROLL_DURATION_MS = 200  # Pre-roll buffering
pre_roll_frames = int(PRE_ROLL_DURATION_MS / CHUNK_DURATION_MS)

AUTHORIZED_USERS: List[str] = [u.nick_name for u in get_users_voice_recognition()]

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

@dataclass
class AiAgentMessage:
    nickname: str
    message: str
    location: str

def filter_authorized_users(transcriptions: List[dict], device: Device) -> List[AiAgentMessage]:
    filtered_text = []
    # Check the line starts with <nickname>:
    for key, value in transcriptions.items():
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

    ring_buffer = collections.deque(maxlen=pre_roll_frames)
    triggered = False
    voiced_frames = []

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
                # Generate and stream TTS audio back to the client
                tts_stream = await generate_tts_stream(message)
                logger.info("Streaming TTS audio to client")
                async for chunk in tts_stream:
                    if chunk:
                        await websocket.send(chunk)
                logger.info("Completed streaming TTS audio to client")
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
                is_speech = vad.is_speech(chunk, MIC_RATE)

                if not triggered:
                    ring_buffer.append((chunk, is_speech))
                    num_voiced = len([f for f, speech in ring_buffer if speech])
                    if num_voiced > VOICED_THRESHOLD * ring_buffer.maxlen:
                        triggered = True
                        logger.info("Speech detected, start recording")
                        # Include pre-roll frames
                        voiced_frames.extend([f for f, s in ring_buffer])
                        ring_buffer.clear()
                else:
                    voiced_frames.append(chunk)
                    ring_buffer.append((chunk, is_speech))
                    num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                    if num_unvoiced > VOICED_THRESHOLD * ring_buffer.maxlen:
                        logger.info("Silence detected, stop recording")
                        # Process the collected audio
                        audio_data = b''.join(voiced_frames)

                        # Save audio data to a WAV file
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                            with wave.open(temp_audio_file, "wb") as wf:
                                wf.setnchannels(CHANNELS)
                                wf.setsampwidth(2)  # 16-bit PCM
                                wf.setframerate(MIC_RATE)
                                wf.writeframes(audio_data)
                            audio_file_path = temp_audio_file.name

                        try:
                            # Send audio file over WebSocket to STT service
                            logger.info("Sending audio data to STT service")
                            with open(audio_file_path, "rb") as f:
                                await stt_socket.send(f.read())
                            transcription = await stt_socket.recv()
                            transcription = json.loads(transcription)
                            logger.info(f"Received transcription: {transcription}")

                            if not transcription:
                                logger.warning("No speech detected in audio. Skipping processing.")
                                # Reset variables for next detection
                                voiced_frames.clear()
                                ring_buffer.clear()
                                triggered = False
                                continue

                            # Filter out unauthorized users
                            filtered: List[AiAgentMessage] = filter_authorized_users(transcription, device=device)
                            if len(filtered) != 0:
                                json_data = json.dumps([asdict(message) for message in filtered])
                                # Send transcription to AI agent
                                await ai_agent_socket.send(json_data)
                            else:
                                logger.info("Transcription is empty, not sending to AI agent")
                        finally:
                            # Ensure the temporary file is removed
                            if os.path.exists(audio_file_path):
                                os.remove(audio_file_path)
                                logger.info(f"Temporary file {audio_file_path} removed")

                        # Reset variables for next detection
                        voiced_frames.clear()
                        ring_buffer.clear()
                        triggered = False
            else:
                logger.warning("Received non-bytes message from client")
    except websockets.ConnectionClosed:
        logger.info(f"Connection closed: {websocket.remote_address}")
    except Exception as e:
        logger.error(f"Error handling client {websocket.remote_address}: {e}")
    finally:
        logger.info(f"Client disconnected: {websocket.remote_address}")
        await stt_socket.close()
        await ai_agent_socket.close()
        ai_listener_task.cancel()

async def main():
    server = await websockets.serve(handle_client, '0.0.0.0', 9001)
    logger.info("Server started on port 9001")
    await server.wait_closed()

if __name__ == "__main__":

    asyncio.run(main())

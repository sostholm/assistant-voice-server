import psycopg
import os
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DATABASE_ADDRESS = os.getenv("DATABASE_ADDRESS", "192.168.0.218")
DATABASE_NAME = os.getenv("DATABASE_NAME", "assistant_v2")
DATABASE_PORT = os.getenv("DATABASE_PORT", "5432")

DSN = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{DATABASE_ADDRESS}:{DATABASE_PORT}/{DATABASE_NAME}"
)
# Connect to your PostgreSQL database
conn = psycopg.connect(
    dbname=DATABASE_NAME,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    host=DATABASE_ADDRESS,
    port=DATABASE_PORT
)


@dataclass
class UserVoiceRecognition:
    user_id: str
    nick_name: Optional[str]

def get_users_voice_recognition() -> List[UserVoiceRecognition]:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            u.user_id,
            u.nick_name
        FROM users u
    """)
    rows = cursor.fetchall()

    cursor.execute("""
        SELECT
            ai.ai_id,
            ai.ai_name
        FROM ai 
    """)

    rows += cursor.fetchall()
    cursor.close()

    results = []
    for row in rows:
        user_id, nick_name, voice_data = row
        if voice_data is None:
            continue
        results.append(UserVoiceRecognition(
            user_id=user_id,
            nick_name=nick_name,
            voice_recognition=voice_data
        ))
    return results


@dataclass
class Device:
    id: int
    device_name: str
    device_type_id: int
    unique_identifier: str
    ip_address: str
    mac_address: str
    location: str
    status: str
    registered_at: datetime
    last_seen_at: datetime

async def get_device_by_id(
    conn: psycopg.AsyncConnection,
    device_id: int
) -> Device:
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, device_name, device_type_id, unique_identifier, ip_address, mac_address, location, status, registered_at, last_seen_at
                FROM devices
                WHERE id = %s
                """,
                (device_id,)
            )
            row = await cur.fetchone()
            if row:
                return Device(*row)
            else:
                return None

    except psycopg.Error as e:
        print("Error occurred while fetching the device.")
        print(e)
        return None
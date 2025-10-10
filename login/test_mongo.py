import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import asyncio

load_dotenv()  # .env 파일 불러오기

async def test_connection():
    uri = os.getenv("MONGO_URI")
    client = AsyncIOMotorClient(uri)
    db = client["ssu_login"]
    print("✅ MongoDB 연결 성공!")
    print("📁 현재 데이터베이스 목록:", await client.list_database_names())

asyncio.run(test_connection())

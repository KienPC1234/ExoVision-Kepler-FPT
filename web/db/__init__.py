from .base import Base, initiate
from .wrapper import AsyncDBWrapper, DBWrapper

USE_ASYNC_DB = False

# ------------------------
# 1️⃣ Khởi tạo engine trước
# ------------------------
initiate(USE_ASYNC_DB)

from .base import engine, SessionLocal, isAsyncEngine

# ------------------------
# 2️⃣ Tạo bảng sau khi engine đã sẵn sàng
# ------------------------
def init_db():
    if engine is None:
        raise RuntimeError("Engine not initialized! Call initiate() first.")

    if USE_ASYNC_DB:
        import asyncio

        async def async_create():
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

        asyncio.run(async_create())
    else:
        Base.metadata.create_all(bind=engine)

# Gọi init_db
init_db()

# ------------------------
# 3️⃣ Khởi tạo connect_db
# ------------------------
if USE_ASYNC_DB:
    connect_db = lambda: AsyncDBWrapper(SessionLocal())
else:
    connect_db = lambda: DBWrapper(SessionLocal())

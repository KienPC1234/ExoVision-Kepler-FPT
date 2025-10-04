from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# ======================
# DATABASE URL
# ======================
DATABASE_URL_SYNC = "postgresql+psycopg2://superuser:Dukuma6Chi7Bolali@localhost:5432/kepler_app_db"
DATABASE_URL_ASYNC = "postgresql+asyncpg://superuser:Dukuma6Chi7Bolali@localhost:5432/kepler_app_db"

# ======================
# GLOBALS
# ======================
engine = None          # Engine object (sync or async)
SessionLocal = None    # sessionmaker object
isAsyncEngine = False  # Flag to know if engine is async

# ======================
# INIT FUNCTION
# ======================
def initiate(use_async: bool = False):
    """
    Initialize the database engine and session.
    :param use_async: True -> async engine, False -> sync engine
    """
    global engine, SessionLocal, isAsyncEngine

    isAsyncEngine = use_async

    if isAsyncEngine:
        engine = create_async_engine(DATABASE_URL_ASYNC, echo=True, future=True)
        SessionLocal = sessionmaker(
            bind=engine, 
            class_=AsyncSession, 
            expire_on_commit=False
        )
    else:
        engine = create_engine(DATABASE_URL_SYNC, echo=True, future=True)
        SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )

# ======================
# BASE CLASS
# ======================
@as_declarative()
class Base:
    id: Any
    __name__: str

    @declared_attr
    def __tablename__(cls) -> str:
        """
        Auto-generate table name as lowercase class name.
        """
        return cls.__name__.lower()

# ======================
# UTILITY FUNCTION
# ======================
def create_all_tables():
    """
    Create all tables defined in Base subclasses.
    Works only for sync engine!
    Async engine must use run_sync(Base.metadata.create_all)
    """
    if engine is None:
        raise RuntimeError("Engine is not initialized. Call `initiate()` first.")

    if isAsyncEngine:
        import asyncio
        async def async_create():
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        asyncio.run(async_create())
    else:
        Base.metadata.create_all(bind=engine)

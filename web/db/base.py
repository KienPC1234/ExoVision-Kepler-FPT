import os
from typing import Any, Optional

from sqlalchemy import create_engine, inspect
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session


DEBUG_FLAG = os.getenv('DEBUG', 'false').lower() == 'true'

# Constants
DATABASE_URL_SYNC = "postgresql+psycopg2://superuser:Dukuma6Chi7Bolali@localhost:5432/kepler_app_db"
DATABASE_URL_ASYNC = "postgresql+asyncpg://superuser:Dukuma6Chi7Bolali@localhost:5432/kepler_app_db"

# ======================
# GLOBALS
# ======================
engine = None          # Engine object (sync or async)
SessionLocal = None    # sessionmaker object
isAsyncEngine = False  # Flag to know if engine is async


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


    if DEBUG_FLAG:
        with SessionLocal() as session:
            create_debug_user(session)

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


def create_debug_user(session: Session) -> Optional[Base]:
    """
    Creates a test user if it doesn't exist and DEBUG_FLAG is True
    Returns the user object if created/found, None otherwise
    """
    if not DEBUG_FLAG:
        return None
        
    # Import User model here to avoid circular imports
    from .models.users import User
    from ..utils.authorizer import hash_pwd
    
    test_user = {
        'email': 'test',
        'hashed_password': hash_pwd('test123')
    }
    
    # Check if test user exists
    existing_user = session.query(User).filter_by(
        email=test_user['email']
    ).first()
    
    if existing_user:
        return existing_user
    
    new_user = User(**test_user)
    session.add(new_user)
    session.commit()
    return new_user


def create_all_tables():
    """
    Create all tables defined in Base subclasses and initialize debug user if needed.
    """
    if engine is None:
        raise RuntimeError("Engine is not initialized. Call `initiate()` first.")

    if isAsyncEngine:
        import asyncio
        async def async_create():
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            # Create debug user in async context
            async with SessionLocal() as session:
                await create_debug_user(session)
        asyncio.run(async_create())
    else:
        Base.metadata.create_all(bind=engine)
        # Create debug user in sync context
        with SessionLocal() as session:
            create_debug_user(session)

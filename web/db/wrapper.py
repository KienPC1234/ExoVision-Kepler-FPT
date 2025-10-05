from typing import Optional, Any, List, Collection, TypeVar, Type, Union
from datetime import datetime, timezone

from sqlalchemy import select, Select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session 

from .base import Base
from .models import User, PredictRecord


T = TypeVar("T")
class DBWrapper:
    def __init__(self, session: Session, auto_commit: bool = True):
        self.session = session

        # Short-circuiting the session methods
        self.commit = session.commit
        self.execute = session.execute
        self.close = session.close
        self.refresh = session.refresh
        self.add = session.add
        self.delete = session.delete  # Add delete method

        # Options
        self.auto_commit = auto_commit

    def flush(self, model: Base) -> None:
        """
        Flush changes to the database (synchronous).
        """
        # If you need to flush only specific object, you can call session.flush()
        # which will flush pending changes to the DB.
        self.session.flush()

    # General save method
    def save_model(self, model: T) -> T:
        """
        Save a model to the database (synchronous).
        """
        self.add(model)
        if self.auto_commit:
            self.commit()
            # refresh the instance from DB after commit
            self.refresh(model)
        return model

    create_user = save_model
    save_user = create_user

    # General get method
    def query_model(self, model_t: Type[T], filters: Collection[Any] | None = None) -> Optional[T]:
        """
        Return a single model instance matching filters, or None.
        """
        query = select(model_t)
        if filters:
            query = query.filter(*filters)
        result = self.execute(query)
        return result.scalar()
    
    def query_model_ex(self, model_t: Type[T], **kwargs) -> Optional[T]:
        return self.query_model(model_t, (getattr(model_t, key) == val for key, val in kwargs.items()))

    def query_models(
        self,
        model_t: Type[T],
        skip: int = 0,
        limit: int = 100,
        filters: Collection[Any] | None = None,
    ) -> list[T]:
        """
        Return a list of model instances.
        """
        query = select(model_t)
        if filters:
            query = query.filter(*filters)
        result = self.execute(query.offset(skip).limit(limit))
        return result.scalars().all()

    # User-related methods
    def get_user(self, username: str) -> Optional[User]:
        return self.query_model_ex(User, email=username)

    def is_username_taken(self, username: str, exclude_user_id: Optional[int] = None) -> bool:
        """
        Check if a username is already taken, excluding a specific user ID.
        """
        filters = (User.email == username,)
        if exclude_user_id:
            filters += (User.id != exclude_user_id,)
        return self.query_model(User, filters=filters) is not None

    def soft_delete_user(self, user: User) -> None:
        """
        Soft delete a user by marking them as deleted and deactivating them.
        Assumes user.soft_delete() is synchronous.
        """
        user.soft_delete()
        self.save_user(user)

    def restore_user(self, user: User) -> None:
        """
        Restore a soft-deleted user.
        Assumes user.soft_restore() is synchronous.
        """
        user.soft_restore()
        self.save_user(user)

    def delete_user(self, user: User) -> None:
        """
        Permanently delete a user instance.
        """
        self.delete(user)
        if self.auto_commit:
            self.commit()

    # History-related methods
    def add_prediction_record(
        self,
        user: Union[User, int],
        type: str,
        name: str,
        result_markdown: str,
        user_data_path: Optional[str] = None,
        output_filename: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> PredictRecord:
        """Create a new prediction record for the user"""
        user_id = user.id if isinstance(user, User) else user
        
        record = PredictRecord(
            user_id=user_id,
            type=type,
            name=name,
            result_markdown=result_markdown,
            user_data_path=user_data_path,
            output_filename=output_filename,
            timestamp=timestamp or datetime.now(timezone.utc),
            has_output_file=bool(output_filename)
        )
        
        return self.save_model(record)


class AsyncDBWrapper:
    def __init__(self, session: AsyncSession, auto_commit: bool = True):
        self.session = session

        # Short-circuiting the session methods
        self.commit = session.commit
        self.execute = session.execute
        self.close = session.close
        self.refresh = session.refresh
        self.add = session.add
        self.delete = session.delete

        # Options
        self.auto_commit = auto_commit

    async def flush(self, model: Any):
        """
        Flush changes to the database.
        """
        await self.session.flush()

    # General save method
    async def save_model(self, model: T) -> T:
        """
        Save a model to the database.
        """
        self.add(model)
        if self.auto_commit:
            await self.commit()
            await self.refresh(model)
        return model

    create_user = save_model
    save_user = create_user

    # General get method
    async def query_model(self, model_t: Type[T], filters: Optional[Collection[Any]] = None) -> Optional[T]:
        query = select(model_t)
        if filters:
            query = query.filter(*filters)
        result = await self.execute(query)
        return result.scalar()

    async def query_model_ex(self, model_t: Type[T], **kwargs) -> Optional[T]:
        return await self.query_model(model_t, (getattr(model_t, key) == val for key, val in kwargs.items()))

    async def query_models(
        self,
        model_t: Type[T],
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Collection[Any]] = None,
    ) -> list[T]:
        query = select(model_t)
        if filters:
            query = query.filter(*filters)
        result = await self.execute(query.offset(skip).limit(limit))
        return result.scalars().all()

    # User-related methods
    async def get_user(self, username: str):
        return await self.query_model_ex(User, email=username)

    async def is_username_taken(self, username: str, exclude_user_id: Optional[int] = None) -> bool:
        filters = User.email == username,
        if exclude_user_id:
            filters += User.id != exclude_user_id,
        return (await self.query_model(User, filters=filters)) is not None

    async def soft_delete_user(self, user: User):
        await user.soft_delete()
        await self.save_user(user)

    async def restore_user(self, user: User):
        await user.soft_restore()
        await self.save_user(user)

    async def delete_user(self, user):
        self.delete(user)
        if self.auto_commit:
            await self.commit()

    # History-related methods
    async def add_prediction_record(
        self,
        user: Union[User, int],
        type: str,
        name: str,
        result_markdown: str,
        user_data_path: Optional[str] = None,
        output_filename: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        new_predict_record = PredictRecord(
            user_id=user.id if isinstance(user, User) else user,
            timestamp=timestamp or datetime.now(timezone.utc),
            type=type,
            name=name,
            result_markdown=result_markdown,
            has_output_file=bool(output_filename),
            user_data_path=user_data_path,
            output_filename=output_filename
        )

        await self.save_model(new_predict_record)
        return new_predict_record
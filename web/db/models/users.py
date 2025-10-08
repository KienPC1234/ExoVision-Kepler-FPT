from sqlalchemy import Boolean, Column, Integer, String, DateTime, JSON
from sqlalchemy.orm import Mapped, relationship
from sqlalchemy.sql import func

from ..base import Base


USER_DEFAULT_CONFIG = {
    "lang": "en",
}

def gen_security_stamp_caller():
    global gen_security_stamp
    if gen_security_stamp is gen_security_stamp_caller:
        from ...utils.authorizer import gen_security_stamp as gss
        gen_security_stamp = gss
    return gen_security_stamp()

gen_security_stamp = gen_security_stamp_caller

def hash_pwd_caller():
    global hash_pwd
    if hash_pwd is hash_pwd_caller:
        from ...utils.authorizer import hash_pwd as hp
        hash_pwd = hp
    return hash_pwd()

hash_pwd = hash_pwd_caller


class User(Base):
    __tablename__: str = "users"

    id: Mapped[int] = Column(Integer, primary_key=True, index=True)
    email: Mapped[str] = Column(String, unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = Column(String, nullable=False)
    security_stamp: Mapped[str] = Column(String, nullable=False, default=gen_security_stamp)
    # preferences: Mapped[dict] = Column(JSON, nullable=False, default=USER_DEFAULT_CONFIG)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_deleted = Column(Boolean, default=False)
    predictions = relationship("PredictRecord", back_populates="user")

    @property
    def is_active(self):
        return not self.is_deleted

    def update_security_stamp(self):
        """
        Updates the security stamp for the user.
        Should be called on security-sensitive changes (e.g., password change).
        """
        self.security_stamp = gen_security_stamp()

    def update_password(self, plain_password: str):
        """
        Updates the user's password and triggers a security stamp update.
        """
        new_hashed_password = hash_pwd(plain_password)

        if self.hashed_password != new_hashed_password:
            self.hashed_password = new_hashed_password

            self.update_security_stamp()

    def soft_delete(self):
        if not self.is_deleted:
            self.is_deleted = True
            self.update_security_stamp()

    def soft_restore(self):
        if self.is_deleted:
            self.is_deleted = False
            self.update_security_stamp()
from sqlalchemy import Boolean, Column, Integer, String, DateTime
from sqlalchemy.orm import Mapped
from sqlalchemy.sql import func


from ..base import Base


class User(Base):
    __tablename__: str = "users"

    id: Mapped[int] = Column(Integer, primary_key=True, index=True)
    email: Mapped[str] = Column(String, unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = Column(String, nullable=False)
    security_stamp: Mapped[str] = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_deleted = Column(Boolean, default=False)


    @property
    def username(self):
        """ The intended login name for users is their email. This is a temporary alias for future considerations. """
        return self.email

    @username.setter
    def username_setter(self, u):
        self.email = u

    @property
    def is_active(self):
        return not self.is_deleted

    def update_security_stamp(self):
        from web.utils.authorizer import gen_security_stamp
        """
        Updates the security stamp for the user.
        Should be called on security-sensitive changes (e.g., password change).
        """
        self.security_stamp = gen_security_stamp()

    def update_password(self, plain_password: str):
        from ...utils.authorizer import hash_pwd
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
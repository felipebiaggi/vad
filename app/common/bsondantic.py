# -*- coding: utf-8 -*-
"""Pydantic objects with bson serialization."""
from __future__ import (
    annotations,
)

from pydantic import (
    BaseModel,
    Extra,
)
import bson


class BsonModel(BaseModel):
    """
    Pydantic base model with serialization/deserialization methods for bson.

    https://docs.mongodb.com/manual/reference/bson-types/

    Should be subclassed the same way BaseModel is subclassed in regular Pydantic.
    """

    def to_bson(
        self,
    ) -> bytes:
        """Serialize object to bson bytes."""
        return bson.encode(self.dict())

    @classmethod
    def from_bson(
        cls,
        bson_bytes: bytes,
    ) -> BsonModel:
        """Parse bson into a BsonModel."""
        return cls(**bson.decode(bson_bytes))

    class Config:
        """Config for the BaseModel."""

        extra = Extra.forbid  # Forbids extra fields.

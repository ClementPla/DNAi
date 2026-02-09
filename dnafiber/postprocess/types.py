from enum import StrEnum


class FiberType(StrEnum):
    ONE_SEGMENT = "one segment"
    TWO_SEGMENTS = "two segments"
    ONE_TWO_ONE = "termination"
    TWO_ONE_TWO = "origin"
    OTHER = "other"

from enum import Enum


class Language(str, Enum):
  
    ENGLISH = "english"
    URDU = "urdu"
    PUNJABI = "punjabi"
    SARAIKI = "saraiki"
    UNKNOWN = "unknown"
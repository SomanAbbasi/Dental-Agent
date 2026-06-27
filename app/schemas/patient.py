import re
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class PatientInfo(BaseModel):

    name: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=100,
        description="Full name of the patient",
    )
    phone: Optional[str] = Field(
        default=None,
        description="Pakistani phone number",
    )
    symptoms: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=500,
        description="Reason for visit / symptoms described by patient",
    )
    time_window: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=200,
        description="Requested appointment time e.g. 'Tuesday morning'",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        # Strip extra whitespace
        v = " ".join(v.split())
        # Block suspiciously short or numeric-only names
        if v.replace(" ", "").isdigit():
            raise ValueError("Name cannot be numeric only")
        # Block injection attempts in name field
        injection_patterns = ["ignore", "system:", "assistant:", "<", ">", "{", "}"]
        v_lower = v.lower()
        for pattern in injection_patterns:
            if pattern in v_lower:
                raise ValueError(f"Invalid characters in name field")
        return v

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        # Strip spaces and dashes for normalization
        cleaned = re.sub(r"[\s\-\(\)]", "", v)
        # Pakistani formats: 03001234567 or +923001234567
        pk_mobile = re.compile(r"^(\+92|0092|0)?3[0-9]{9}$")
        if not pk_mobile.match(cleaned):
            raise ValueError(
                "Phone must be a valid Pakistani mobile number "
                "e.g. 03001234567 or +923001234567"
            )
        return cleaned

    @field_validator("symptoms")
    @classmethod
    def validate_symptoms(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        # Block attempts to inject clinical advice requests into symptoms
        clinical_injection = [
            "prescribe", "diagnose", "what medicine", "which antibiotic",
            "ignore previous", "ignore above", "new instruction",
        ]
        v_lower = v.lower()
        for pattern in clinical_injection:
            if pattern in v_lower:
                raise ValueError(
                    "Symptoms field contains disallowed clinical request patterns"
                )
        return v

    def is_complete(self) -> bool:
        return all([
            self.name is not None,
            self.phone is not None,
            self.symptoms is not None,
            self.time_window is not None,
        ])

    def missing_fields(self) -> list[str]:
        missing = []
        if self.name is None:
            missing.append("name")
        if self.phone is None:
            missing.append("phone")
        if self.symptoms is None:
            missing.append("symptoms")
        if self.time_window is None:
            missing.append("time_window")
        return missing
    
    
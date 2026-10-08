from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class LdapStatusResponse(BaseModel):
    enabled: bool
    configured: bool
    uri: Optional[str] = None
    base_dn: Optional[str] = None
    start_tls: bool = False
    insecure_skip_tls_verify: bool = False
    #: "ldaps", "starttls" or "plaintext".
    transport: str = "plaintext"
    verifies_certificates: bool = False
    has_service_account: bool = False
    role_mapping_configured: bool = False
    group_search_configured: bool = False
    create_user_on_login: bool = True
    local_fallback_when_unavailable: bool = False
    #: Why the settings cannot work, in words; empty when they can.
    problems: list[str] = []


class LdapTestRequest(BaseModel):
    username: Optional[str] = Field(
        None,
        max_length=255,
        description="A login name to look up in the directory (no password).",
    )


class LdapTestStep(BaseModel):
    step: str
    ok: bool
    message: str


class LdapTestUser(BaseModel):
    username: str
    dn: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    groups: list[str] = []
    #: The role the group mapping would give; None when no mapping is set.
    role: Optional[str] = None


class LdapTestResponse(BaseModel):
    ok: bool
    steps: list[LdapTestStep] = []
    user: Optional[LdapTestUser] = None


class LdapImportRequest(BaseModel):
    search_filter: Optional[str] = Field(
        None,
        description="LDAP search filter; defaults to LDAP_IMPORT_FILTER",
    )
    limit: int = Field(200, ge=1, le=5000)
    dry_run: bool = True
    default_role: str = Field("user", pattern="^(admin|user|viewer)$")
    overwrite_role: bool = False


class LdapImportUserRow(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    dn: Optional[str] = None
    role: str
    action: str  # created, updated, skipped, error
    error: Optional[str] = None


class LdapImportResponse(BaseModel):
    created: int
    updated: int
    skipped: int
    errors: int
    rows: list[LdapImportUserRow] = []

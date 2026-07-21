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


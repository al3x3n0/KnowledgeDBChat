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
    #: The attribute accounts are matched by; empty means by DN.
    unique_id_attribute: Optional[str] = None
    #: Minutes between scheduled syncs; 0 means they are off.
    sync_interval_minutes: int = 0
    #: The last sync's report, scheduled or run from this page.
    last_sync: Optional["LdapSyncResponse"] = None


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
    #: created, updated, refused or error.
    action: str
    error: Optional[str] = None


class LdapImportResponse(BaseModel):
    created: int
    updated: int
    #: Entries matching a local account that is not provably the same person.
    refused: int = 0
    skipped: int
    errors: int
    rows: list[LdapImportUserRow] = []


class LdapSyncRequest(BaseModel):
    #: Say what would change without changing it. On by default: a sync
    #: deactivates accounts.
    dry_run: bool = True


class LdapSyncResponse(BaseModel):
    ran_at: str
    dry_run: bool
    directory_entries: int = 0
    managed_accounts: int = 0
    unchanged: int = 0
    updated: list[str] = []
    role_changes: list[str] = []
    not_found: list[str] = []
    deactivated: list[str] = []
    reactivated: list[str] = []
    held_back: Optional[str] = None
    error: Optional[str] = None


class LdapVouchRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=255)
    #: False withdraws an earlier vouch.
    verified: bool = True


class LdapVouchResponse(BaseModel):
    username: str
    email: str
    is_verified: bool
    #: Whether the account was made from the sign-in page: the vouch is what
    #: lets the directory link such an account by its email.
    self_registered: bool


LdapStatusResponse.model_rebuild()

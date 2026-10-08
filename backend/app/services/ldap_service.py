"""
LDAP / Active Directory: authenticating users and importing them.

Uses ``ldap3`` (pure Python). Everything here is synchronous network I/O;
callers on the event loop use the ``*_async`` wrappers.

What this module is careful about, each because the first version was not:

* **The server's certificate is verified.** ``ldap3`` validates nothing unless
  handed a ``Tls`` object that says to, so ``ldaps://`` and StartTLS accepted
  any certificate and the passwords went to whoever answered -- with
  ``LDAP_INSECURE_SKIP_TLS_VERIFY`` at its default of false.
* **A password is not sent in the clear** unless ``LDAP_ALLOW_PLAINTEXT`` says
  so: plain ``ldap://`` without StartTLS is refused.
* **The username is escaped** before it goes into a search filter or a DN.
  It was interpolated with ``str.format``, so ``*`` matched any user and
  ``)(`` rewrote the filter.
* **"Wrong password" and "directory unreachable" are different answers.**
  ``authenticate`` returns None for the first and raises ``LdapUnavailable``
  for the second. They used to both be None, and the caller's fallback to a
  local password -- meant for an outage -- ran on every rejected login.
* **A request has a time limit** in both directions, not only to connect.
"""

from __future__ import annotations

import asyncio
import re
import ssl
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from loguru import logger

from app.core.config import settings

#: Active Directory's paged-results control.
_PAGED_RESULTS_OID = "1.2.840.113556.1.4.319"


class LdapError(Exception):
    """Something about LDAP that is not the user's credentials."""


class LdapUnavailable(LdapError):
    """The directory could not be asked: unreachable, TLS failed, timed out,
    or the service account was refused."""


class LdapMisconfigured(LdapError):
    """The settings cannot work as written; the message says which."""


@dataclass(frozen=True)
class LdapUser:
    username: str
    dn: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    groups: list[str] = field(default_factory=list)
    raw: dict[str, Any] | None = None


def _split_csv(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def split_dns(value: Optional[str]) -> list[str]:
    """A setting holding several DNs, separated by ``;`` or newlines.

    Not commas: a DN is itself a comma-separated list. These settings were
    documented as "comma-separated group DNs" and split on commas, which cut
    every DN into its RDNs -- ``cn=admins``, ``ou=groups``, ... -- none of
    which is a group. Role mapping by group could never match.
    """
    return [
        part.strip() for part in re.split(r"[;\n]", str(value or "")) if part.strip()
    ]


def normalize_dn(dn: str) -> str:
    """A DN in a form two spellings of the same entry share.

    Directories return ``CN=Admins,OU=Groups,DC=ex,DC=com``; people configure
    ``cn=admins, ou=groups, dc=ex, dc=com``. Compared as strings those differ,
    and a role mapping that never matches fails silently as "user".
    """
    parts = [part.strip() for part in str(dn or "").split(",")]
    return ",".join(
        "=".join(piece.strip() for piece in part.split("=", 1)) for part in parts
    ).lower()


def _setting(name: str, default: Any = None) -> Any:
    return getattr(settings, name, default)


def _text(name: str) -> str:
    return str(_setting(name) or "").strip()


class LdapService:
    """Reads settings when asked, so a changed setting needs no new instance."""

    # -- configuration ------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(_setting("LDAP_ENABLED", False))

    @property
    def uri(self) -> str:
        return _text("LDAP_URI")

    @property
    def base_dn(self) -> str:
        return _text("LDAP_BASE_DN")

    def is_configured(self) -> bool:
        return bool(self.enabled and self.uri and self.base_dn)

    def transport(self) -> str:
        """``ldaps``, ``starttls`` or ``plaintext``."""
        if self.uri.lower().startswith("ldaps://"):
            return "ldaps"
        if bool(_setting("LDAP_START_TLS", False)):
            return "starttls"
        return "plaintext"

    def verifies_certificates(self) -> bool:
        return self.transport() != "plaintext" and not bool(
            _setting("LDAP_INSECURE_SKIP_TLS_VERIFY", False)
        )

    def role_mapping_configured(self) -> bool:
        return bool(_text("LDAP_ADMIN_GROUP_DNS") or _text("LDAP_VIEWER_GROUP_DNS"))

    def problems(self) -> list[str]:
        """Why the current settings cannot work, in words an admin can act on."""
        if not self.enabled:
            return []
        found = []
        if not self.uri:
            found.append("LDAP_URI is not set.")
        if not self.base_dn:
            found.append("LDAP_BASE_DN is not set.")
        if self.transport() == "plaintext" and not bool(
            _setting("LDAP_ALLOW_PLAINTEXT", False)
        ):
            found.append(
                "The connection is not encrypted: passwords would cross the "
                "network in the clear. Use an ldaps:// URI or LDAP_START_TLS=true "
                "(or LDAP_ALLOW_PLAINTEXT=true to accept that)."
            )
        template = _text("LDAP_USER_DN_TEMPLATE")
        if template and "{username}" not in template:
            found.append("LDAP_USER_DN_TEMPLATE has no {username} placeholder.")
        if not template:
            if "{username}" not in _text("LDAP_USER_SEARCH_FILTER"):
                found.append("LDAP_USER_SEARCH_FILTER has no {username} placeholder.")
            if not _text("LDAP_BIND_DN"):
                found.append(
                    "Neither LDAP_USER_DN_TEMPLATE nor a service account "
                    "(LDAP_BIND_DN) is set, so a user's entry cannot be found."
                )
        return found

    def _require_usable(self) -> None:
        if not self.is_configured():
            raise LdapMisconfigured("LDAP is not enabled and configured.")
        problems = self.problems()
        if problems:
            raise LdapMisconfigured(" ".join(problems))

    # -- connections --------------------------------------------------------

    def _ldap3(self):
        try:
            import ldap3  # type: ignore

            return ldap3
        except Exception as e:  # pragma: no cover - a broken install
            raise LdapMisconfigured(f"ldap3 is required for LDAP support: {e}")

    def _tls(self):
        """How the server's certificate is checked; None for plaintext."""
        if self.transport() == "plaintext":
            return None
        ldap3 = self._ldap3()
        if bool(_setting("LDAP_INSECURE_SKIP_TLS_VERIFY", False)):
            return ldap3.Tls(validate=ssl.CERT_NONE)
        # Said explicitly: with no Tls object ldap3 validates nothing.
        return ldap3.Tls(
            validate=ssl.CERT_REQUIRED,
            ca_certs_file=_text("LDAP_CA_CERT_FILE") or None,
        )

    def _make_server(self):
        ldap3 = self._ldap3()
        return ldap3.Server(
            self.uri,
            use_ssl=self.transport() == "ldaps",
            get_info=ldap3.NONE,
            tls=self._tls(),
            connect_timeout=int(_setting("LDAP_CONNECT_TIMEOUT_SECONDS", 8)),
        )

    def _new_connection(self, bind_dn: Optional[str], bind_password: Optional[str]):
        """An unopened connection. The seam tests replace with a mock server."""
        ldap3 = self._ldap3()
        return ldap3.Connection(
            self._make_server(),
            user=bind_dn,
            password=bind_password,
            auto_bind=False,
            raise_exceptions=True,
            receive_timeout=int(_setting("LDAP_RECEIVE_TIMEOUT_SECONDS", 10)),
        )

    def _connect(self, bind_dn: Optional[str], bind_password: Optional[str]):
        """Open, secure and bind. Returns the bound connection, or None when
        the server refused these credentials; raises LdapUnavailable for
        everything that is not about the credentials."""
        from ldap3.core import exceptions as ldap_errors

        self._require_usable()
        try:
            conn = self._new_connection(bind_dn, bind_password)
            conn.open()
            if self.transport() == "starttls":
                conn.start_tls()
            if conn.bind():
                return conn
            conn.unbind()
            return None
        except ldap_errors.LDAPInvalidCredentialsResult:
            return None
        except ldap_errors.LDAPPasswordIsMandatoryError:
            return None
        except ldap_errors.LDAPException as exc:
            raise LdapUnavailable(f"{type(exc).__name__}: {exc}") from exc
        except (OSError, ssl.SSLError) as exc:
            raise LdapUnavailable(f"{type(exc).__name__}: {exc}") from exc

    def _service_connection(self):
        """Bound as the service account. Its failure is ours, not a user's."""
        bind_dn = _text("LDAP_BIND_DN") or None
        bind_password = _text("LDAP_BIND_PASSWORD") or None
        conn = self._connect(bind_dn, bind_password)
        if conn is None:
            raise LdapUnavailable(
                "The service account (LDAP_BIND_DN) was refused by the directory."
            )
        return conn

    # -- reading entries ----------------------------------------------------

    def _attrs(self) -> list[str]:
        attrs = _split_csv(_setting("LDAP_USER_ATTRIBUTES", ""))
        return attrs or ["uid", "mail", "cn", "displayName", "memberOf"]

    def _extract(
        self,
        attrs: dict[str, Any],
        *,
        fallback_username: str,
        dn: str,
        extra_groups: Iterable[str] = (),
    ) -> LdapUser:
        def _first(attr: str) -> Optional[str]:
            v = attrs.get(attr)
            if v is None:
                return None
            if isinstance(v, (list, tuple)):
                return str(v[0]) if v else None
            return str(v)

        username = _first(_text("LDAP_USERNAME_ATTRIBUTE") or "uid") or (
            fallback_username
        )
        email = _first(_text("LDAP_EMAIL_ATTRIBUTE") or "mail")
        full_name = _first(_text("LDAP_FULL_NAME_ATTRIBUTE") or "displayName") or (
            _first("cn")
        )

        groups: list[str] = []
        gv = attrs.get(_text("LDAP_GROUPS_ATTRIBUTE") or "memberOf")
        if isinstance(gv, (list, tuple)):
            groups = [str(x) for x in gv if x]
        elif isinstance(gv, str) and gv:
            groups = [gv]
        seen = {normalize_dn(g) for g in groups}
        for group in extra_groups:
            if normalize_dn(group) not in seen:
                seen.add(normalize_dn(group))
                groups.append(str(group))

        if not email:
            domain = _text("LDAP_DEFAULT_EMAIL_DOMAIN")
            if domain and username:
                email = f"{username}@{domain}"

        return LdapUser(
            username=username,
            dn=dn,
            email=email,
            full_name=full_name,
            groups=groups,
            raw=dict(attrs),
        )

    def _search_user(self, conn, username: str) -> tuple[str, dict[str, Any]] | None:
        """The one entry the user filter matches for this name, or None."""
        ldap3 = self._ldap3()
        from ldap3.utils.conv import escape_filter_chars

        search_filter = _text("LDAP_USER_SEARCH_FILTER").format(
            username=escape_filter_chars(username)
        )
        conn.search(
            search_base=self.base_dn,
            search_filter=search_filter,
            search_scope=ldap3.SUBTREE,
            attributes=self._attrs(),
            size_limit=2,
        )
        entries = list(conn.entries or [])
        if not entries:
            return None
        if len(entries) > 1:
            # Two people answering to one login name is not a login.
            logger.warning(
                f"LDAP: {len(entries)} entries match the login name {username!r}; "
                "refusing to guess."
            )
            return None
        entry = entries[0]
        return str(entry.entry_dn), dict(entry.entry_attributes_as_dict or {})

    def _user_dn_from_template(self, username: str) -> Optional[str]:
        template = _text("LDAP_USER_DN_TEMPLATE")
        if not template:
            return None
        from ldap3.utils.dn import escape_rdn

        return template.format(username=escape_rdn(username))

    def _group_search(self, conn, *, user_dn: str, username: str) -> list[str]:
        """Groups found by searching for the user as a member.

        ``memberOf`` on the user is an Active Directory convenience; OpenLDAP
        has it only with an overlay. With ``LDAP_GROUP_SEARCH_BASE`` set the
        groups are looked up the other way round, which also allows AD's
        nested-membership rule in the filter.
        """
        base = _text("LDAP_GROUP_SEARCH_BASE")
        if not base:
            return []
        ldap3 = self._ldap3()
        from ldap3.utils.conv import escape_filter_chars

        search_filter = _text("LDAP_GROUP_SEARCH_FILTER").format(
            user_dn=escape_filter_chars(user_dn),
            username=escape_filter_chars(username),
        )
        try:
            conn.search(
                search_base=base,
                search_filter=search_filter,
                search_scope=ldap3.SUBTREE,
                attributes=["cn"],
                size_limit=500,
            )
        except Exception as exc:
            logger.warning(f"LDAP group search failed for {username!r}: {exc}")
            return []
        return [str(entry.entry_dn) for entry in (conn.entries or [])]

    # -- operations ---------------------------------------------------------

    def authenticate(self, username: str, password: str) -> Optional[LdapUser]:
        """The directory's entry for this user if the password is theirs.

        None means the directory answered and said no: unknown user or wrong
        password. :class:`LdapUnavailable` means it could not be asked, which
        is the only case a caller may treat as an outage.
        """
        self._require_usable()
        username = (username or "").strip()
        # A bind with a DN and no password is an *anonymous* bind on many
        # servers, and succeeds. ldap3 refuses the empty string; a password
        # of spaces is refused here.
        if not username or not (password or "").strip() or "\x00" in username:
            return None

        user_dn = self._user_dn_from_template(username)
        attrs: dict[str, Any] = {}
        groups: list[str] = []

        if user_dn is None:
            service = self._service_connection()
            try:
                found = self._search_user(service, username)
                if not found:
                    return None
                user_dn, attrs = found
                groups = self._group_search(service, user_dn=user_dn, username=username)
            finally:
                service.unbind()

        conn = self._connect(user_dn, password)
        if conn is None:
            logger.info(f"LDAP: the directory refused the password for {username!r}.")
            return None
        try:
            if not attrs:
                # Template mode: read the entry as the user themselves.
                found = self._search_user(conn, username)
                if found:
                    _dn, attrs = found
                groups = self._group_search(conn, user_dn=user_dn, username=username)
        finally:
            conn.unbind()

        return self._extract(
            attrs, fallback_username=username, dn=user_dn, extra_groups=groups
        )

    def find_user(self, username: str) -> Optional[LdapUser]:
        """The directory's entry for a login name, without a password.

        Needs the service account. For the admin diagnostic: "what would the
        directory tell us about this person?"
        """
        self._require_usable()
        if not _text("LDAP_BIND_DN"):
            raise LdapMisconfigured(
                "Looking a user up without their password needs LDAP_BIND_DN."
            )
        service = self._service_connection()
        try:
            found = self._search_user(service, (username or "").strip())
            if not found:
                return None
            dn, attrs = found
            groups = self._group_search(service, user_dn=dn, username=username)
        finally:
            service.unbind()
        return self._extract(
            attrs, fallback_username=username, dn=dn, extra_groups=groups
        )

    def search_users(self, *, search_filter: str, limit: int = 200) -> list[LdapUser]:
        """Directory users matching an admin-supplied filter (no passwords)."""
        self._require_usable()
        if not _text("LDAP_BIND_DN"):
            raise LdapMisconfigured("LDAP_BIND_DN is required for user import/search.")

        ldap3 = self._ldap3()
        out: list[LdapUser] = []
        page_size = int(_setting("LDAP_SEARCH_PAGE_SIZE", 200))
        cookie = None

        conn = self._service_connection()
        try:
            while True:
                conn.search(
                    search_base=self.base_dn,
                    search_filter=search_filter,
                    search_scope=ldap3.SUBTREE,
                    attributes=self._attrs(),
                    paged_size=min(page_size, max(1, limit)),
                    paged_cookie=cookie,
                )
                for entry in conn.entries or []:
                    if len(out) >= limit:
                        return out
                    out.append(
                        self._extract(
                            dict(entry.entry_attributes_as_dict or {}),
                            fallback_username="",
                            dn=str(entry.entry_dn),
                        )
                    )
                cookie = (
                    ((conn.result or {}).get("controls") or {})
                    .get(_PAGED_RESULTS_OID, {})
                    .get("value", {})
                    .get("cookie")
                )
                if not cookie:
                    break
        finally:
            conn.unbind()
        return out

    def map_role(self, groups: Iterable[str] | None) -> Optional[str]:
        """The role the configured group mapping gives, or None if there is none.

        None is "no mapping configured, leave the role alone". It used to be
        "user", so a local admin who signed in through LDAP on a deployment
        that mapped no groups was demoted on the spot.
        """
        if not self.role_mapping_configured():
            return None
        member_of = {normalize_dn(g) for g in (groups or []) if g}
        admins = {normalize_dn(g) for g in split_dns(_text("LDAP_ADMIN_GROUP_DNS"))}
        viewers = {normalize_dn(g) for g in split_dns(_text("LDAP_VIEWER_GROUP_DNS"))}
        if member_of & admins:
            return "admin"
        if member_of & viewers:
            return "viewer"
        return "user"

    def diagnose(self, username: Optional[str] = None) -> dict[str, Any]:
        """Walk the configuration the way a login would, and say where it stops.

        Never raises and never needs a user's password: each step reports
        ``ok`` and a message, so an admin setting LDAP up sees "the
        certificate could not be verified" rather than a login that fails.
        """
        steps: list[dict[str, Any]] = []

        def step(name: str, ok: bool, message: str) -> bool:
            steps.append({"step": name, "ok": ok, "message": message})
            return ok

        report: dict[str, Any] = {"ok": False, "steps": steps, "user": None}

        if not step(
            "enabled",
            self.enabled,
            "LDAP_ENABLED is on." if self.enabled else "LDAP_ENABLED is off.",
        ):
            return report
        problems = self.problems()
        if not step(
            "settings",
            not problems,
            " ".join(problems) or "The settings are consistent.",
        ):
            return report
        step(
            "transport",
            True,
            {
                "ldaps": "LDAPS (TLS from the first byte).",
                "starttls": "StartTLS on a plain connection.",
                "plaintext": "Unencrypted, allowed by LDAP_ALLOW_PLAINTEXT.",
            }[self.transport()]
            + (
                ""
                if self.transport() == "plaintext"
                else (
                    " The server's certificate is verified."
                    if self.verifies_certificates()
                    else " The server's certificate is NOT verified "
                    "(LDAP_INSECURE_SKIP_TLS_VERIFY)."
                )
            ),
        )

        service_dn = _text("LDAP_BIND_DN")
        try:
            if service_dn:
                conn = self._service_connection()
                conn.unbind()
                step("service_account", True, f"Bound as {service_dn}.")
            else:
                # No service account: at least prove the server answers.
                conn = self._new_connection(None, None)
                try:
                    conn.open()
                    if self.transport() == "starttls":
                        conn.start_tls()
                finally:
                    conn.unbind()
                step(
                    "service_account",
                    True,
                    "No service account; the server accepted the connection. "
                    "Users are bound directly from LDAP_USER_DN_TEMPLATE.",
                )
        except LdapError as exc:
            step("service_account", False, str(exc))
            return report
        except Exception as exc:
            step("service_account", False, f"{type(exc).__name__}: {exc}")
            return report

        if username and service_dn:
            try:
                user = self.find_user(username)
            except LdapError as exc:
                step("user_lookup", False, str(exc))
                return report
            if user is None:
                step(
                    "user_lookup",
                    False,
                    f"No single entry under {self.base_dn} matches {username!r} "
                    "with LDAP_USER_SEARCH_FILTER.",
                )
                return report
            role = self.map_role(user.groups)
            step(
                "user_lookup",
                True,
                f"Found {user.dn}; {len(user.groups)} group(s); "
                + (
                    f"would be given the role '{role}'."
                    if role
                    else "no group mapping is configured, so the role is left as it is."
                ),
            )
            if not user.email:
                step(
                    "email",
                    False,
                    "The entry has no email and LDAP_DEFAULT_EMAIL_DOMAIN is not "
                    "set, so an account could not be created for this user.",
                )
                return report
            report["user"] = {
                "username": user.username,
                "dn": user.dn,
                "email": user.email,
                "full_name": user.full_name,
                "groups": user.groups,
                "role": role,
            }
        elif username:
            step(
                "user_lookup",
                True,
                "Skipped: looking a user up without their password needs a "
                "service account (LDAP_BIND_DN).",
            )

        report["ok"] = True
        return report

    # -- for callers on the event loop --------------------------------------

    async def authenticate_async(self, username: str, password: str):
        return await asyncio.to_thread(self.authenticate, username, password)

    async def search_users_async(self, *, search_filter: str, limit: int = 200):
        return await asyncio.to_thread(
            self.search_users, search_filter=search_filter, limit=limit
        )

    async def diagnose_async(self, username: Optional[str] = None):
        return await asyncio.to_thread(self.diagnose, username)


ldap_service = LdapService()

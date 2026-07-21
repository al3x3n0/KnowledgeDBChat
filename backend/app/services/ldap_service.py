"""
LDAP integration for authenticating users and importing directory users.

We use `ldap3` (pure python) to avoid system-level dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Iterable

from loguru import logger

from app.core.config import settings


@dataclass(frozen=True)
class LdapUser:
    username: str
    dn: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    groups: list[str] | None = None
    raw: dict[str, Any] | None = None


def _split_csv(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


class LdapService:
    def __init__(self) -> None:
        self.enabled = bool(getattr(settings, "LDAP_ENABLED", False))
        self.uri = (getattr(settings, "LDAP_URI", None) or "").strip()
        self.base_dn = (getattr(settings, "LDAP_BASE_DN", None) or "").strip()

    def is_configured(self) -> bool:
        if not self.enabled:
            return False
        return bool(self.uri and self.base_dn)

    def _ldap3(self):
        try:
            import ldap3  # type: ignore

            return ldap3
        except Exception as e:
            raise RuntimeError(f"ldap3 is required for LDAP support: {e}")

    def _make_server(self):
        ldap3 = self._ldap3()

        # ldap3's Tls verification knobs are limited; for most setups, relying on system CA works.
        tls = None
        if getattr(settings, "LDAP_INSECURE_SKIP_TLS_VERIFY", False):
            import ssl

            tls = ldap3.Tls(validate=ssl.CERT_NONE)

        return ldap3.Server(
            self.uri,
            use_ssl=self.uri.lower().startswith("ldaps://"),
            get_info=ldap3.NONE,
            tls=tls,
            connect_timeout=int(getattr(settings, "LDAP_CONNECT_TIMEOUT_SECONDS", 8)),
        )

    def _connect(self, bind_dn: Optional[str], bind_password: Optional[str]):
        ldap3 = self._ldap3()
        server = self._make_server()
        conn = ldap3.Connection(
            server,
            user=bind_dn,
            password=bind_password,
            auto_bind=False,
            raise_exceptions=True,
        )
        conn.open()
        if getattr(settings, "LDAP_START_TLS", False) and not self.uri.lower().startswith("ldaps://"):
            conn.start_tls()
        conn.bind()
        return conn

    def _attrs(self) -> list[str]:
        raw = getattr(settings, "LDAP_USER_ATTRIBUTES", "") or ""
        attrs = [x.strip() for x in raw.split(",") if x.strip()]
        return attrs or ["uid", "mail", "cn", "displayName", "memberOf"]

    def _extract(self, attrs: dict[str, Any], *, fallback_username: str, dn: str) -> LdapUser:
        def _first(attr: str) -> Optional[str]:
            v = attrs.get(attr)
            if v is None:
                return None
            if isinstance(v, (list, tuple)):
                return str(v[0]) if v else None
            return str(v)

        username_attr = getattr(settings, "LDAP_USERNAME_ATTRIBUTE", "uid") or "uid"
        email_attr = getattr(settings, "LDAP_EMAIL_ATTRIBUTE", "mail") or "mail"
        name_attr = getattr(settings, "LDAP_FULL_NAME_ATTRIBUTE", "displayName") or "displayName"
        groups_attr = getattr(settings, "LDAP_GROUPS_ATTRIBUTE", "memberOf") or "memberOf"

        username = _first(username_attr) or fallback_username
        email = _first(email_attr)
        full_name = _first(name_attr) or _first("cn")

        groups: list[str] = []
        gv = attrs.get(groups_attr)
        if isinstance(gv, (list, tuple)):
            groups = [str(x) for x in gv if x]
        elif isinstance(gv, str):
            groups = [gv]

        if not email:
            domain = (getattr(settings, "LDAP_DEFAULT_EMAIL_DOMAIN", None) or "").strip()
            if domain:
                email = f"{username}@{domain}"

        return LdapUser(
            username=username,
            dn=dn,
            email=email,
            full_name=full_name,
            groups=groups,
            raw=attrs,
        )

    def _search_user_dn(self, conn, username: str) -> tuple[str, dict[str, Any]] | None:
        ldap3 = self._ldap3()

        filt_tmpl = getattr(settings, "LDAP_USER_SEARCH_FILTER", "") or ""
        filt = filt_tmpl.format(username=username)
        attrs = self._attrs()

        conn.search(
            search_base=self.base_dn,
            search_filter=filt,
            search_scope=ldap3.SUBTREE,
            attributes=attrs,
            size_limit=1,
        )
        if not conn.entries:
            return None

        entry = conn.entries[0]
        dn = str(entry.entry_dn)
        data = entry.entry_attributes_as_dict or {}
        return dn, data

    def authenticate_and_fetch(self, username: str, password: str) -> Optional[LdapUser]:
        """
        Validate username/password against LDAP. Returns user attributes on success.
        """
        if not self.is_configured():
            return None

        # 1) Determine user DN
        user_dn = None
        user_attrs: dict[str, Any] = {}

        dn_tmpl = (getattr(settings, "LDAP_USER_DN_TEMPLATE", None) or "").strip()
        if dn_tmpl:
            user_dn = dn_tmpl.format(username=username)
        else:
            bind_dn = (getattr(settings, "LDAP_BIND_DN", None) or "").strip() or None
            bind_pw = (getattr(settings, "LDAP_BIND_PASSWORD", None) or "").strip() or None
            try:
                with self._connect(bind_dn, bind_pw) as conn:
                    found = self._search_user_dn(conn, username)
                    if not found:
                        return None
                    user_dn, user_attrs = found
            except Exception as e:
                logger.warning(f"LDAP search failed: {e}")
                return None

        if not user_dn:
            return None

        # 2) Bind as the user to verify password
        try:
            with self._connect(user_dn, password) as conn:
                # If we didn't fetch attributes during the search, fetch them now.
                if not user_attrs:
                    found = self._search_user_dn(conn, username)
                    if found:
                        _dn, user_attrs = found
                return self._extract(user_attrs or {}, fallback_username=username, dn=user_dn)
        except Exception as e:
            logger.info(f"LDAP bind failed for {username}: {e}")
            return None

    def search_users(self, *, search_filter: str, limit: int = 200) -> list[LdapUser]:
        """
        Search LDAP and return user entries (no password auth).
        """
        if not self.is_configured():
            return []

        bind_dn = (getattr(settings, "LDAP_BIND_DN", None) or "").strip() or None
        bind_pw = (getattr(settings, "LDAP_BIND_PASSWORD", None) or "").strip() or None
        if not bind_dn:
            raise RuntimeError("LDAP_BIND_DN is required for user import/search")

        ldap3 = self._ldap3()
        attrs = self._attrs()
        out: list[LdapUser] = []

        page_size = int(getattr(settings, "LDAP_SEARCH_PAGE_SIZE", 200))
        cookie = None

        with self._connect(bind_dn, bind_pw) as conn:
            while True:
                conn.search(
                    search_base=self.base_dn,
                    search_filter=search_filter,
                    search_scope=ldap3.SUBTREE,
                    attributes=attrs,
                    paged_size=min(page_size, max(1, limit)),
                    paged_cookie=cookie,
                )
                for entry in conn.entries:
                    if len(out) >= limit:
                        return out
                    dn = str(entry.entry_dn)
                    data = entry.entry_attributes_as_dict or {}
                    # fallback_username not meaningful here; best effort from attr
                    out.append(self._extract(data, fallback_username="", dn=dn))

                cookie = conn.result.get("controls", {}).get("1.2.840.113556.1.4.319", {}).get("value", {}).get("cookie")
                if not cookie:
                    break

        return out

    def map_role(self, groups: Iterable[str] | None) -> str:
        groups_set = {g.strip().lower() for g in (groups or []) if g}
        admin_dns = {g.lower() for g in _split_csv(getattr(settings, "LDAP_ADMIN_GROUP_DNS", None))}
        viewer_dns = {g.lower() for g in _split_csv(getattr(settings, "LDAP_VIEWER_GROUP_DNS", None))}

        if admin_dns and (groups_set & admin_dns):
            return "admin"
        if viewer_dns and (groups_set & viewer_dns):
            return "viewer"
        return "user"


ldap_service = LdapService()


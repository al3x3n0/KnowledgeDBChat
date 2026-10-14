"""The in-memory directory the LDAP tests share, and its fixtures.

Registered in ``conftest.py`` so a test file uses ``directory`` and ``ldap``
without importing them; an imported fixture reads to flake8 as a name
redefined at every use.
"""

import pytest
from ldap3 import MOCK_SYNC, OFFLINE_SLAPD_2_4, Connection, Server
from ldap3.core.exceptions import LDAPSocketOpenError

from app.core.config import settings
from app.services import ldap_service as ldap_module
from app.services.ldap_service import LdapService

BASE = "dc=example,dc=com"
SERVICE_DN = f"cn=svc,{BASE}"
ALICE_DN = f"uid=alice,ou=People,{BASE}"
ADMINS_DN = f"cn=admins,ou=Groups,{BASE}"


class Directory:
    """One in-memory LDAP server that every connection in a test shares."""

    def __init__(self):
        self.server = Server("mock", get_info=OFFLINE_SLAPD_2_4)
        self.down = False
        self.binds = []
        seed = self._connection(None, None)
        self.add = seed.strategy.add_entry
        self.add(
            SERVICE_DN,
            {"userPassword": "svc-secret", "objectClass": "person", "sn": "svc"},
        )
        self.add(
            ALICE_DN,
            {
                "userPassword": "alice-secret",
                "objectClass": "inetOrgPerson",
                "uid": "alice",
                "mail": "alice@example.com",
                "cn": "Alice Adams",
                "sn": "Adams",
                "displayName": "Alice Adams",
            },
        )
        self.add(
            f"uid=bob,ou=People,{BASE}",
            {
                "userPassword": "bob-secret",
                "objectClass": "inetOrgPerson",
                "uid": "bob",
                "mail": "bob@example.com",
                "cn": "Bob",
                "sn": "B",
            },
        )
        self.add(
            ADMINS_DN,
            {"objectClass": "groupOfNames", "cn": "admins", "member": [ALICE_DN]},
        )

    def _connection(self, bind_dn, password):
        return Connection(
            self.server,
            user=bind_dn,
            password=password,
            client_strategy=MOCK_SYNC,
            raise_exceptions=True,
        )

    def new_connection(self, bind_dn, password):
        if self.down:
            raise LDAPSocketOpenError("connection refused")
        self.binds.append(bind_dn)
        return self._connection(bind_dn, password)


@pytest.fixture
def directory(monkeypatch):
    directory = Directory()
    for name, value in {
        "LDAP_ENABLED": True,
        "LDAP_URI": "ldaps://directory.example.com",
        "LDAP_BASE_DN": BASE,
        "LDAP_BIND_DN": SERVICE_DN,
        "LDAP_BIND_PASSWORD": "svc-secret",
        "LDAP_USER_DN_TEMPLATE": None,
        "LDAP_USER_SEARCH_FILTER": "(uid={username})",
        "LDAP_USER_ATTRIBUTES": "uid,mail,cn,displayName,memberOf",
        "LDAP_START_TLS": False,
        "LDAP_ALLOW_PLAINTEXT": False,
        "LDAP_INSECURE_SKIP_TLS_VERIFY": False,
        "LDAP_CA_CERT_FILE": None,
        "LDAP_ADMIN_GROUP_DNS": None,
        "LDAP_VIEWER_GROUP_DNS": None,
        "LDAP_GROUP_SEARCH_BASE": None,
        "LDAP_DEFAULT_EMAIL_DOMAIN": None,
        "LDAP_CREATE_USER_ON_LOGIN": True,
        "LDAP_SYNC_ON_LOGIN": True,
        "LDAP_LOCAL_FALLBACK_WHEN_UNAVAILABLE": False,
        "LDAP_LINK_LOCAL_USERS_BY_USERNAME": False,
    }.items():
        monkeypatch.setattr(settings, name, value, raising=False)

    def _new_connection(self, bind_dn, password):
        return directory.new_connection(bind_dn, password)

    monkeypatch.setattr(LdapService, "_new_connection", _new_connection)
    return directory


@pytest.fixture
def ldap():
    return ldap_module.ldap_service

"""Keeps a visitor's own endpoint from being used to reach the server's network.

The visitor's URL is fetched by this server, so an unchecked one makes the app an open proxy
into its own network (loopback services, cloud metadata at 169.254.169.254, private ranges).
`check_url` rejects a bad address when it is entered; `guarded_client` checks again before every
request, because a DNS name can resolve to a public address at entry and a private one later.
Redirects are never followed, as a public server could redirect to a private one.

Operator endpoints from `Settings` are trusted and do not pass through here.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Final
from urllib.parse import urlsplit

import httpx2  # the openai SDK's HTTP layer

HTTPS_PORT: Final = 443
NOT_HTTPS: Final = "A custom endpoint must start with https://, so the key is sent encrypted."
NO_HOST: Final = "That address has no host name. Use a full URL such as https://api.example.com/v1."
USERINFO: Final = (
    "Leave the user name and password out of the address; enter the key in its own box."
)
NOT_443: Final = "A custom endpoint must use the standard https port (443)."
INSIDE: Final = (
    "That address points inside the server's network ({ip}), so it can't be used. "
    "Use a public https endpoint, such as a hosted provider or a tunnel to your own machine."
)
UNRESOLVED: Final = "The host name {host} could not be found. Check the address."


class BlockedEndpointError(ValueError):
    """The address is one the server must not connect to; the message says why, without the key."""


def check_url(url: str, *, allow_private: bool) -> None:
    """Reject a URL that must not be used as a visitor's endpoint.

    With `allow_private` (local testing) http, other ports and private hosts are accepted.

    Raises:
        BlockedEndpointError: If the URL is malformed or points somewhere forbidden.
    """
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise BlockedEndpointError(NO_HOST) from None
    if not allow_private and parts.scheme != "https":
        raise BlockedEndpointError(NOT_HTTPS)
    if not parts.hostname:
        raise BlockedEndpointError(NO_HOST)
    if "@" in parts.netloc:
        raise BlockedEndpointError(USERINFO)
    if allow_private:
        return
    if port not in (None, HTTPS_PORT):
        raise BlockedEndpointError(NOT_443)
    check_host(parts.hostname, HTTPS_PORT)


def check_host(host: str, port: int) -> None:
    """Reject `host` if any address it resolves to is not on the public internet.

    Raises:
        BlockedEndpointError: If the name does not resolve or resolves to a forbidden address.
    """
    try:
        found = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise BlockedEndpointError(UNRESOLVED.format(host=host)) from None
    for *_, sockaddr in found:
        ip = ipaddress.ip_address(str(sockaddr[0]).split("%")[0])  # drop an IPv6 zone id
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
            or not ip.is_global
        ):
            raise BlockedEndpointError(INSIDE.format(ip=ip))


class _VettingTransport(httpx2.HTTPTransport):
    """An HTTP transport that vets the destination again before every request."""

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        check_host(request.url.host, request.url.port or HTTPS_PORT)
        return super().handle_request(request)


def guarded_client() -> httpx2.Client:
    """Return an HTTP client for a visitor's endpoint: no redirects, vetted on every request."""
    return httpx2.Client(transport=_VettingTransport(), follow_redirects=False)

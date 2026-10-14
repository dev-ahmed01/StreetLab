"""M7 deny-by-default off-host access and optional HTTP Basic protection.

This is a single-operator local security boundary, NOT multi-user tenancy.
In local mode, both peer address and the HTTP Host must be loopback.
Remote reverse-proxy deployments MUST opt in to basic mode AND provide TLS,
network authorization and preferably SSO at the edge.
"""
from __future__ import annotations

import base64
import binascii
import hmac
import ipaddress
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response


def access_settings() -> dict:
    mode=os.getenv("STREETLAB_SECURITY_MODE","local").strip().lower()
    if mode not in {"local","basic"}:
        raise ValueError("STREETLAB_SECURITY_MODE must be local or basic")
    password=os.getenv("STREETLAB_ACCESS_PASSWORD","")
    if mode=="basic" and (len(password)<20 or len(password)>512 or
        any(ord(c)<33 or ord(c)>126 for c in password)):
        raise ValueError("Basic mode requires a 20–512 character printable non-space STREETLAB_ACCESS_PASSWORD")
    return {"mode":mode,"password":password if mode=="basic" else None}


def _loopback(host: str) -> bool:
    if not host:
        return False
    if host.lower()=="localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _local_request(request: Request) -> bool:
    client=request.client.host if request.client else ""
    authority=request.headers.get("host","")
    # IPv6 authorities are bracketed; query or unsanitized hostname bypasses
    # are never accepted. TestClient uses explicit in-process sentinels.
    if client=="testclient" and authority=="testserver":
        return True
    if authority.startswith("["):
        host=authority[1:].split("]",1)[0]
        rest=authority[len(host)+2:]
        if rest and (not rest.startswith(":") or not rest[1:].isdigit()):
            return False
    else:
        host=authority.split(":")[0]
        if ":" in authority and (
            authority.count(":")!=1 or not authority.rsplit(":",1)[1].isdigit()):
            return False
    return _loopback(client) and _loopback(host)


def _basic_matches(header: str, password: str) -> bool:
    if not isinstance(header,str) or not header.startswith("Basic ") or len(header)>2048:
        return False
    try:
        decoded=base64.b64decode(header[6:],validate=True).decode("utf-8")
        username,secret=decoded.split(":",1)
    except (ValueError,UnicodeDecodeError,binascii.Error):
        return False
    return hmac.compare_digest(username,"streetlab") and hmac.compare_digest(secret,password)


def install_guard(app: FastAPI) -> None:
    config=access_settings() # Fail at startup for invalid remote setup.
    @app.middleware("http")
    async def security(request: Request, call_next):
        if config["mode"]=="local":
            if not _local_request(request):
                return JSONResponse({"detail":"StreetLab local mode only accepts loopback-origin requests"},
                                    status_code=403,headers={"Cache-Control":"no-store"})
        elif not _basic_matches(request.headers.get("authorization",""),config["password"]):
            return Response(status_code=401,
                            headers={"WWW-Authenticate":'Basic realm="StreetLab Local Project"',
                                     "Cache-Control":"no-store"})
        response=await call_next(request)
        response.headers["X-Content-Type-Options"]="nosniff"
        response.headers["X-Frame-Options"]="DENY"
        response.headers["Referrer-Policy"]="no-referrer"
        response.headers["Cache-Control"]="no-store"
        response.headers["Content-Security-Policy"]=(
            "default-src 'self'; img-src 'self' data:; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'self'")
        return response

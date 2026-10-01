"""The User-Agent the SEC requires from every automated request.

EDGAR's fair access policy asks each client to declare who it is and how to
reach them, in the form ``Sample Company Name AdminContact@sample.com``. A
request without one is refused with 403, and a placeholder shared by everyone
who copied the same template gets that template blocked for all of them. So the
value is checked before the first request, not discovered by the SEC.
"""

import re

MAX_USER_AGENT_CHARS = 200
EMAIL_PATTERN = re.compile(r"[^@\s]+@([^@\s]+\.[^@\s]+|localhost)", re.ASCII)
# Domains that can never receive mail (RFC 2606 and RFC 6761)
RESERVED_DOMAINS = ("example.com", "example.net", "example.org")
RESERVED_SUFFIXES = (".example", ".test", ".invalid", ".localhost")
PLACEHOLDERS = ("your.email", "your name", "youremail")


def validate_user_agent(user_agent: str) -> str:
    """Return the trimmed User-Agent, or raise if it does not declare a sender.

    A declared sender is printable ASCII (it travels as an HTTP header), carries
    one e-mail address on a domain that can receive mail, and names someone
    besides that address.

    Raises:
        ValueError: when the value is missing, a template placeholder, or not
            something the SEC could use to contact the sender.
    """
    value = user_agent.strip()
    if len(value) > MAX_USER_AGENT_CHARS:
        raise ValueError(f"the User-Agent is longer than {MAX_USER_AGENT_CHARS} characters")
    if not value.isascii() or not value.isprintable():
        raise ValueError("the User-Agent must be printable ASCII on one line")

    lowered = value.lower()
    if any(placeholder in lowered for placeholder in PLACEHOLDERS):
        raise ValueError("the User-Agent is still the template placeholder")

    emails = list(EMAIL_PATTERN.finditer(value))
    if len(emails) != 1:
        raise ValueError("the User-Agent must carry exactly one contact e-mail address")
    email = emails[0]
    if _is_reserved(email.group(1).lower()):
        raise ValueError(f"{email.group(1)} cannot receive mail; use a real contact address")
    if not value.replace(email.group(0), "").strip():
        raise ValueError("the User-Agent must name the sender as well as the address")
    return value


def _is_reserved(domain: str) -> bool:
    if domain == "localhost" or domain.endswith(RESERVED_SUFFIXES):
        return True
    return any(
        domain == reserved or domain.endswith(f".{reserved}") for reserved in RESERVED_DOMAINS
    )

# Security policy

## Supported versions

Only the `main` branch is supported. Fixes land there; there are no maintained
release branches.

## Scope

edgar-rag is a service meant to run on your own machine, next to a local Ollama.
The threat model follows from that:

- **Filing text is untrusted input.** Anyone can file a document with the SEC, and
  exhibits carry third-party text. Passages are treated as data, never as
  instructions, and anything in them that could impersonate the prompt, fabricate
  a citation or force a refusal is neutralised before the model sees it. A way
  around that is a vulnerability.
- **There is no authentication, by design.** The service binds to a local address
  and is not meant to be exposed to a network. Running it on a public interface
  is outside the supported setup.
- **Secrets stay out of the repository.** The only credential-like value is the
  SEC User-Agent (a contact e-mail address), read from the environment or a local
  `.env` file that git ignores.

In scope: prompt injection through filing text, citations that point at text the
answer did not come from, error responses that leak internal URLs or stack traces,
unbounded downloads or requests to EDGAR, and loading an index file that executes
code.

Out of scope: attacks that require control of the local Ollama server or of the
machine running the service.

## Reporting a vulnerability

Please do not open a public issue. E-mail **email.rodrigopalma@gmail.com** with a
description, the steps to reproduce it and the commit you tested against. You can
expect an acknowledgement within 7 days.

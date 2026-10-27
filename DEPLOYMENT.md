# Deployment guide

This repository provides a single-host deployment package for the public-facing
Streamlit app, Ollama and a persistent Chroma index. The web page exposes a
chat with cited answers and collapsible source passages. Conversation turns
are held in the current Streamlit session. Model choice, PDF path, passage count
and index creation are operator settings. Obvious individual care and dosing
requests are redirected to health professionals; this rule-based guard is not a
complete clinical safety system.

## Before public launch

1. Have Kenyan clinical and public health reviewers approve the PDF version,
   public wording, example answers and escalation message. The citation checker
   checks page numbers; it does **not** establish clinical correctness.
2. Confirm permission to host and reproduce passages from the Kenya Ministry of
   Health PDF. Its copyright page states that rights are reserved. The PDF is
   mounted at runtime and excluded from the application image. Review the
   chosen model's license and terms for public service use as well.
3. Test representative questions, including vague symptoms, suspected cases,
   emergency requests, dosing requests, out-of-scope questions and attempts to
   override the assistant's instructions. Record reviewer sign-off before
   accepting public traffic.
4. Choose a host with enough memory for Ollama, the 8B chat model, the embedding
   model and Chroma. A suitable GPU will improve generation speed. Budget for
   model downloads and the first index build on deployment.
5. Provide HTTPS and edge request rate limits or abuse protection. The optional
   `compose.public.yaml` adds Caddy for automatic HTTPS; use an additional edge
   control for rate limiting. The base Compose file binds Streamlit to the
   host's loopback interface; Ollama has **no** published host port. Do not
   expose Ollama's API directly to the internet.

## Start on a deployment host

Install Docker Engine and the Docker Compose plugin. Place the reviewed guideline
PDF on the host. Copy `.env.example` to `.env`, then set `GUIDELINE_SOURCE` to
the PDF's absolute host path. For example:

```sh
cp .env.example .env
# Edit GUIDELINE_SOURCE in .env before starting.
docker compose config --quiet
docker compose up -d --build
docker compose ps
docker compose logs -f models indexer app
```

The `models` job downloads the two Ollama models, and the `indexer` job builds
the Chroma index before the app starts. Both can take several minutes on a fresh
host. The app listens on `127.0.0.1:8501`. The health endpoint is
`/_stcore/health` on port 8501. To publish through the included Caddy overlay,
set `SITE_DOMAIN` in `.env`, point that domain's DNS to the server, open ports
80 and 443, and run:

```sh
docker compose -f compose.yaml -f compose.public.yaml up -d --build
```

Caddy handles TLS and proxies to the app. Add edge rate limiting and abuse
protection before announcing the service publicly. If the host already has a
managed HTTPS ingress, use only the base Compose file and proxy its loopback
port instead.

For an NVIDIA GPU host with the NVIDIA Container Toolkit installed, use:

```sh
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build
```

The app and index share a named volume; Ollama models use a separate named
volume. Back up both volumes if fast recovery matters. Run one app replica with
this local Chroma store. To change the PDF, stop the stack, update the mounted
file, and start it again so the indexer rebuilds before the app serves requests.

## Operating the public service

- Keep the app behind HTTPS. Apply rate limits at the ingress, and monitor
  availability and response time. `MAX_CONCURRENT_ANSWERS` limits simultaneous
  in-app generations; extra users receive a busy message.
- Avoid collecting patient names, contact details or medical records. Review
  proxy and application logs for accidental question or answer retention.
- Keep the review process active when the Kenya or WHO guidance changes. The
  WHO PDFs in `who_guidelines/` are reference downloads and are **not** indexed
  by this assistant.
- Pin the exact container image digests and Python dependency set used for a
  release, and rebuild after security updates. The supplied Compose file uses
  the current official Ollama image tag as a starting point.

This package prepares a staging deployment. Public launch still requires the
content, copyright, privacy and infrastructure checks above.

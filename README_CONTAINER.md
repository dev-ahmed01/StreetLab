# StreetLab containerized local research workstation

This Docker setup runs the real StreetLab web UI, native SUMO, video ingestion,
SQLite job queue and research evidence viewer. It is an academic software
prototype, not a field-calibrated traffic signal controller.

## Run on Windows Docker Desktop / WSL2

1. Copy .env.example to .env and replace the placeholder with a unique random
   password of at least 20 printable characters. Keep .env private.
2. Create an empty artifacts directory next to compose.yaml.
3. Run: docker compose up --build -d
4. Open http://127.0.0.1:8000 on this computer. HTTP Basic username is
   streetlab and the password is from your local .env.
5. Check: docker compose ps and docker compose logs -f web.
6. Stop safely: docker compose down. Avoid down -v, which erases volume data.

Host port binding is restricted to 127.0.0.1, using HTTP Basic inside Docker
because Docker proxy peers are not the original application loopback address.
Do not expose this version to the internet, a VPS, a tunnel, or a LAN. The
application has no remote multi-user security or TLS configuration.

The app pages are /, /research, /reports, /release, /fieldwork,
/calibration, /baseline and /scenarios.

## Existing real research evidence

The Docker image retains only frozen Git-tracked historical source summaries.
For the completed canonical May25 audit, put the original verified
automated_object_audit.json in artifacts/. For already executed synthetic
native SUMO figures, copy the real streetlab_research_bundle.zip there.
GitHub Actions research-submission.yml provides a validated source-hashed ZIP.
Missing/invalid artifacts deliberately return HTTP 409, not demonstration data.

## Real video inference is optional, not fabricated

The web image genuinely decodes, checksums and stores MP4 uploads, and queues
durable analysis jobs. Real inference is a separate worker, because the original
SHA-pinned W04 OpenVINO model and source provenance are private.

On the operator's own machine, place the matching W04 XML/BIN model tree
in private/w04-model and the verified provenance in
private/w04-provenance.json. Then use:

    docker compose -f compose.yaml -f compose.worker.yaml --profile inference up --build -d

The worker image installs the heavier SAHI/Ultralytics/OpenVINO/tracker stack.
It refuses missing or hash-mismatched frozen model inputs. A generated demo
MP4 only tests the software decoder and queue, not vehicle detection accuracy.

## Automated checks

GitHub workflow container-e2e.yml builds the Docker image and verifies the
actual HTTP server, Basic authorization, pages, frozen science flags, real
decodable MP4 ingestion, duplicate rejection, queue/cancel/retry, database
persistence across a container restart and a native SUMO paired decision
experiment. The original frozen W04 inference and independent site
geometry/census/holdout are still NOT validated by GitHub CI.

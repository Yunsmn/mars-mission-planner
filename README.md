# MARVIN — Mars Autonomous Reasoning & Verification INtelligence

Onboard, offline mission planner for a planetary rover.
**IBM AI Builders Challenge — August (Space Exploration).**

> **The full README is authored by IBM Bob** as part of the build — problem, solution,
> architecture, real-data usage, demo, and the required **"How IBM Bob was used"** section.
> This stub only orients the repo; see [`docs/`](docs/) for the specs Bob builds against:
> [DESIGN](docs/DESIGN.md) · [DATA](docs/DATA.md) · [INTERFACES](docs/INTERFACES.md) ·
> [SENSORS](docs/SENSORS.md) · [Bob's task brief](docs/BUILD_WITH_BOB.md).

## Setup & run

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
.venv/bin/python -m demo.run          # scripted drive-and-grab simulation demo
.venv/bin/python -m pytest            # tests
```

## License

MIT — see [`LICENSE`](LICENSE).

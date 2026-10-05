# Phase 2 review

`palette-proposal.svg` records the approved palette study. `theme-check.md`
exercises code, tabs, callouts, links, buttons, tables, and branding. It stays
outside `docs/` so it is not included in the tutorial or its navigation.

To preview the fixture, activate the docs environment and run this from
`docsite/`. Each invocation creates a new, ignored preview directory, preserving
previous builds:

```bash
python - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import shutil
import yaml

stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
target = Path(".preview") / f"theme-review-{stamp}"
target.mkdir(parents=True)
shutil.copytree("docs", target / "docs")
shutil.copytree("overrides", target / "overrides")
shutil.copyfile("review/theme-check.md", target / "docs/components.md")
config = yaml.safe_load(Path("mkdocs.yml").read_text())
config.update(docs_dir="docs", site_dir="site", nav=[
    {"Home": "index.md"}, {"Theme component review": "components.md"},
])
(target / "mkdocs.yml").write_text(yaml.safe_dump(config, sort_keys=False))
print(f"mkdocs serve --config-file {target}/mkdocs.yml")
PY
```

Run the printed command, then visit <http://127.0.0.1:8000/components/>. The
temporary navigation belongs only to the component preview; it does not propose
the tutorial's information architecture.

## Validation record — 2026-09-23

- Strict MkDocs builds passed for the site and component fixture.
- Chromium 153 via Playwright 1.63.0: desktop 1440×900 and mobile 320×760,
  both schemes. No page-level horizontal overflow or JavaScript errors.
- Dark is the initial scheme; light/dark choices survive reload. Local fonts
  load correctly. Browser checks recorded no external requests after the source
  partial disabled GitHub's repository-statistics fetch.
- Search, the mobile drawer, linked tabs, actual clipboard contents, expandable
  details, visible keyboard focus, and reduced-motion styles were checked.
- Axe-core 4.13.0 found no violations in its WCAG A/AA rules for the home page,
  component fixture, open search, or mobile drawer in the tested states.
  Mobile search produced contrast checks needing manual review because its
  overlay covers page content. Visible search colors were checked separately.
- These checks cover the current placeholder and fixture in Chromium, not a
  complete accessibility certification or cross-browser review. Recheck future
  tutorial pages, screenshots, and any new interactive components.

Screenshots and machine-readable audit results remain in the ignored
`.preview/` directory. They contain only placeholder and fixture content.

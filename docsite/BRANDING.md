# Theme assets and provenance

The approved colors and typography are recorded in [THEME_PROPOSAL.md](THEME_PROPOSAL.md).
All theme assets are served from this directory. No font CDN is required.

## IBM Plex

The unmodified WOFF2 files in `docs/assets/fonts/` come from the official
[IBM Plex repository at commit `763c36e`](https://github.com/IBM/plex/tree/763c36ef9117782905ae010056dfbe8fd2653a25).
Source directories are `packages/plex-sans/fonts/complete/woff2/` and
`packages/plex-mono/fonts/complete/woff2/`. The packages' identical SIL Open Font
License 1.1 is included as [OFL.txt](docs/assets/fonts/OFL.txt).

Sans supplies regular, medium, semibold, bold, and regular italic; Mono supplies
regular, medium, and bold. Fonts load on demand with `font-display: swap`, using
Material's system fallbacks. Together the fonts, license, SVG, and PNG are
484,516 bytes (about 473 KiB).

SHA-256 checksums for the downloaded files:

```text
ba711a3085ff9f27440b6b9c4550cfc47c97bf36591d5da958b975bb3add8c1a  IBMPlexSans-Regular.woff2
5660f8a658f8bb50dbc005232f885eadffd2bc1c235c4f6fbb63469d1f9cde6d  IBMPlexSans-Medium.woff2
f78048030eab62e860efa39a0df79e2e5581bf122eb95b9bc42c0b8a4988d205  IBMPlexSans-SemiBold.woff2
fa7130d854a660b39a7fc9e6e0f2dc23dba5f1346e2adea3e1fe37b6d884133d  IBMPlexSans-Bold.woff2
13284fab1821ba6e3652c1580fcf2bbfd8c9309520c69b3d1224dab40b37c597  IBMPlexSans-Italic.woff2
ba204497f16b6d334cee9d1e963a831b73e3a56e1d6300a8489d18df7214b350  IBMPlexMono-Regular.woff2
33faf307fa6031fb4062276d7320a6d632de890cbb347576fd80cfa01077bc25  IBMPlexMono-Medium.woff2
ea576f38d05cc44cca48c45314984beb8cc1d2b886f58e1dce99f15dc344eb1d  IBMPlexMono-Bold.woff2
91c25c350d3cac39da2736d74f7ba37ef648f5237a4e330a240615bc8d8c4360  OFL.txt
```

## Logo and favicon

`docs/assets/logo.svg` preserves the app's `cF` mark from
`cryofilter/app/static/index.html:17` and `styles.css:72-84` at app commit
`081007c`. Its 26px square, 8px radius, 0.5px border, colors, and IBM Plex Mono
Medium 12px lettering follow the app. Glyphs are paths, so the SVG does not
depend on installed fonts. The glyph outlines are centered vertically.
`docs/assets/favicon.png` is a transparent 64px rasterization of that SVG.
Material renders `cryoFILTER` as the adjacent site wordmark.

[review/build-branding.py](review/build-branding.py) regenerates both derived
assets. It needs the separate tooling environment with `fonttools[woff]==4.65.0`,
`playwright==1.63.0`, and Chromium; these are not documentation build dependencies.
When that environment is present, run from `docsite/`:

```bash
TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" \
XDG_CONFIG_HOME="$PWD/.preview/config" \
PLAYWRIGHT_BROWSERS_PATH="$PWD/.preview/browsers" \
  .browser-env/bin/python review/build-branding.py
```

## Repository link

The small [source partial](overrides/partials/source.html) follows Material
9.7.7's template but omits its repository-statistics JavaScript hook. The private
repository link remains usable, without requesting GitHub statistics on page
load. Material's license is retained in
[overrides/LICENSE.material.txt](overrides/LICENSE.material.txt).

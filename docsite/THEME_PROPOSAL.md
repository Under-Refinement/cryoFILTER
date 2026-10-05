# Phase 2 theme proposal

Status: palette, typography, and implemented theme approved by the maintainer.
Phase 2 is complete. See [review/README.md](review/README.md) for validation.

Source baseline: the integrated cryoFILTER app and tutorial scaffold.
The app has one dark scheme. The light scheme below is an explicitly derived
counterpart, not an existing app theme. Default to dark and remember the reader's
manual light/dark choice. Keep Material's navigation and responsive layout.

The [palette study](review/palette-proposal.svg) compares the proposed colors.
It is a review illustration, not a screenshot of an implemented site.

## Palette for approval

All CSS source line references below refer to
`cryofilter/app/static/styles.css` in the repository root.

| Named docs token | Dark | Light | App source and derivation |
|---|---|---|---|
| Page `--cf-page` | `#0A0F14` | `#F5F8FB` | Dark: `--bg-0`, line 3, used by body. Light: 60% white mixed with app `--fg` from line 10. |
| Panel `--cf-panel` | `#101922` | `#FFFFFF` | Dark: `--bg-2`, line 5, used by cards/forms. White is the mask outline in `cryofilter/qc_render.py:12`, reused as a light surface. |
| Code well `--cf-code` | `#070C10` | `#E6EEF4` | Dark: `--bg-well`, line 7, used by logs. Light: app `--fg`, line 10, reassigned as a surface. |
| Body text `--cf-text` | `#E6EEF4` | `#16222C` | Dark: `--fg`, line 10. Light: app `--bg-3`, line 6, reassigned as text. |
| Secondary text `--cf-muted` | `#8DA0AE` | `#4E6273` | App `--fg-2`, line 11, and `--fg-4`, line 13. Use the latter only on light surfaces. |
| Links/actions `--cf-accent` | `#4FB8D8` | `#1C667D` | App `--accent`, line 14, used by actions and focus. Light: darken its HSL lightness with hue/saturation held fixed. |
| Action text `--cf-on-accent` | `#04222B` | `#FFFFFF` | Dark: `--on-accent`, line 16, used on primary buttons. Light: white on the darker accent. |
| Success `--cf-success` | `#3FBF9B` | `#236955` | App `--ok`, line 18, used by success/clean status. Light: same HSL derivation as accent. |
| Warning `--cf-warning` | `#D9A441` | `#7A5818` | App `--warn`, line 19, used by queued/warning status. Light: same HSL derivation. |
| Error `--cf-error` | `#E57676` | `#B52222` | App `--err`, line 20, is `#E05A5A`. Lighten for dark docs and darken for light docs to preserve text contrast. |

Light accent/status colors reduce the source HSL lightness in steps of 0.005,
holding hue and saturation fixed, until contrast is at least 5.5:1 against all
three light surfaces: `#FFFFFF`, `#F5F8FB`, and `#E6EEF4`. Hex channels are rounded
to the nearest integer. The step counts are accent 56, success 45, warning 53,
and error 39. Dark error increases the source lightness by 13 steps to meet the
same target against the darkest scheme's lightest surface, `#16222C`.

Supporting tokens:

- Header/navigation surface: app `--bg-1` (`#0C131A`, line 4) in dark mode;
  white in light mode. Use the scheme's normal text color on this surface.
- Raised/control surface: app `--bg-3` (`#16222C`) in dark mode; `#E6EEF4` in light.
- Decorative separators: app `--line`/`--line-2` (`#1B2833`/`#2A3B49`, lines 8–9)
  in dark mode. For light mode, `#DCE0E3` mixes 20% app `--fg-4` with 80% white.
- Essential control boundaries: app `--fg-3` (`#61798A`, line 12) in both modes;
  this is distinct from the subtle decorative separators.
- Focus ring: the scheme accent, 2px with a 2px offset, following lines 45–47.
- Documentation magenta, if needed for code highlighting: `#E56FAA` in dark mode
  and `#AF2068` in light. These derive from app `--mask` (`#E0559B`, line 17)
  by increasing HSL lightness 12 steps or decreasing it 40 steps, respectively.
  Scientific images retain their original colors.

## Material role mapping

Use named `cryofilter-dark` and `cryofilter-light` schemes with `primary: custom`
and `accent: custom`. Each token's eventual CSS comment will retain the app
source path/line and any derivation above.

| Material role | Proposed token |
|---|---|
| `--md-default-bg-color` | `--cf-page` |
| `--md-default-fg-color`, `--md-typeset-color` | `--cf-text` |
| `--md-default-fg-color--light` | `--cf-muted` |
| `--md-primary-fg-color`, `--md-accent-fg-color`, `--md-typeset-a-color` | `--cf-accent` |
| `--md-primary-bg-color`, `--md-accent-bg-color` | `--cf-on-accent` |
| `--md-code-bg-color` | `--cf-code` |
| `--md-code-fg-color` | `--cf-text` |
| Code comment color | `--cf-muted` |
| Code keyword/string/number colors | Accent/success/documentation magenta, respectively |
| Note/tip/warning/failure admonition accents | Accent/success/warning/error, respectively |

Material uses its primary color for both links and the header. Keep the primary
role as the accent for readable links; use a small, grouped header/tab surface
override so the header matches the app's neutral chrome. Admonition body/title
text uses the normal text color, with status accents for icons and borders and
a 10% status tint on title backgrounds. This preserves legibility when text
wraps. Retain visible link underlines in prose and a visible keyboard focus ring.

## Contrast checks and changes from the app

Ratios use the WCAG sRGB relative-luminance formula. Normal text must meet 4.5:1;
essential control boundaries must meet 3:1. These calculations validate the
listed pairings; rendered components, opacity, and focus still need browser
checks after implementation.

| Pairing | Dark | Light |
|---|---:|---:|
| Body text on page | 16.40:1 | 15.16:1 |
| Secondary text on raised/control surface | 5.98:1 | 5.39:1 |
| Link on page | 8.41:1 | 6.07:1 |
| Link on raised/control surface | 7.07:1 | 5.51:1 |
| Text on filled action | 7.24:1 | 6.47:1 |
| Essential control boundary on raised surface | 3.55:1 | 3.88:1 |

Resolve three conflicts rather than copying unreadable combinations:

1. App cyan on white is only 2.29:1. Light-mode links and actions use `#1C667D`.
2. App `--fg-3` on a dark card is 3.89:1. Essential secondary text uses `--fg-2`;
   the dimmer source tokens are not copied into readable dark-mode text roles.
3. App red on `--bg-3` is 4.45:1. Dark documentation error text uses `#E57676`
   (5.53:1 on that surface). App rendering and scientific overlays are unchanged.

## Typography, branding, and geometry

- Prose/headings: IBM Plex Sans, with the app's system sans-serif fallbacks
  (`styles.css:33`). Code: IBM Plex Mono, with its monospace fallbacks (line 81).
  Bundle licensed web fonts locally with their license; keep Google font
  autoloading disabled so the site directory remains self-contained.
- Keep Material's prose scale: normally 16px with line height 1.6, and H1/H2/H3
  at 2em/1.5625em/1.25em. Use the app's 500 heading weight. The app's 13px body
  and 14px/16px/14px headings suit compact forms; the tutorial needs a readable
  document hierarchy. Preserve Material's relative sizing and browser zoom.
- Preserve the app's `cF` mark and `cryoFILTER` wordmark from
  `cryofilter/app/static/index.html:17`. Derive an SVG logo and a small PNG
  favicon from that mark after approval.
- Use the app's 8px corner radius (`styles.css:22`) for code/admonition surfaces.
  The app also declares 12px (line 23) but does not currently use it.
- The app uses mostly 0.5px borders, 36px minimum inputs, and a single dialog
  shadow of `0 24px 70px rgba(0,0,0,0.45)` (line 691). Retain Material control
  sizing, use 1px docs borders for clearer edges, and keep surfaces mostly flat.
- Respect reduced motion, preserve visible keyboard focus, provide alt text on
  images, and check both schemes at 320px width after implementation.

## Remaining app tokens and scientific colors

The table and supporting notes account for all declared app CSS colors except
`--accent-dim` (`#1E3A47`, line 15), declared but unused in the current stylesheet,
and `--log-fg` (`#93A6B4`, line 21), used for log text. Keep these recorded as
source colors; do not introduce additional docs roles without a use for them.
The canvas well also uses literal `#05070A` (line 1020), and the connection
backdrop uses `rgba(7,12,16,0.58)` (line 675).

The current cryoFILTER annotation and diagnostic renderers agree on subtype
colors: Carbon `#915794`, Crystalline `#327CA3`, Aggregate `#DE4D25`, and Ethane
`#6F8270`. Sources: `cryofilter/app/static/app.js:44` and `cryofilter/qc_render.py:20`.

Untyped annotation masks use `#C9B7FF` at 25% overlay opacity (`app.js:45`, `:939`),
while diagnostic masks use `#B27ECD` with white outlines (`qc_render.py:11`).
Accepted particles use white circles with `#0072BD` outlines; rejected particles
use black X marks with `#D95319` outlines (`qc_render.py:13`). The probability
colormap is viridis (`qc_render.py:19`). Annotation selection uses the app cyan,
erase uses the original app red, and shape handles use `#F8FBFF` (`app.js:51`,
`:966`). Do not recolor these images to fit the documentation theme.

## Sources and approval boundary

- [Material custom colors and schemes](https://squidfunk.github.io/mkdocs-material/setup/changing-the-colors/#customization)
- [Material local fonts](https://squidfunk.github.io/mkdocs-material/setup/changing-the-fonts/#customization)
- [IBM Plex source and license](https://github.com/IBM/plex)
- [WCAG text contrast](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html)
- [WCAG non-text contrast](https://www.w3.org/WAI/WCAG22/Understanding/non-text-contrast.html)

The build brief requires approval of the palette table before custom CSS is
written. That approval was received before implementation. Phase 2 now includes
the named schemes, local fonts, derived logo/favicon, component styles, and a
review fixture outside the tutorial. The maintainer approved that implementation;
Phase 3's navigation outline is in [INFORMATION_ARCHITECTURE.md](INFORMATION_ARCHITECTURE.md).

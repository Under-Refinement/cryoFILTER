"""Regenerate the app-derived SVG mark and 64px PNG favicon, from docsite.

Tooling only: fonttools[woff]==4.65.0 and playwright==1.63.0 with Chromium.
These packages are not needed to build or serve the documentation.
"""

import asyncio
from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from playwright.async_api import async_playwright


ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets"


async def main():
    font = TTFont(ASSETS / "fonts" / "IBMPlexMono-Medium.woff2")
    glyphs = font.getGlyphSet()
    scale = 12 / font["head"].unitsPerEm
    advance = 0
    paths = []
    bounds = []
    for character in "cF":
        glyph = glyphs[font.getBestCmap()[ord(character)]]
        path_pen = SVGPathPen(glyphs)
        bounds_pen = BoundsPen(glyphs)
        glyph.draw(path_pen)
        glyph.draw(bounds_pen)
        paths.append(f'<path transform="translate({advance} 0)" d="{path_pen.getCommands()}"/>')
        bounds.append(bounds_pen.bounds)
        advance += glyph.width

    # App styles.css:72-84: 26px square, radius 8px, 0.5px line, 12px Mono 500.
    # Center the outlined glyphs vertically; use the font's advances horizontally.
    x = (26 - advance * scale) / 2
    y = 13 + (min(b[1] for b in bounds) + max(b[3] for b in bounds)) * scale / 2
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 26 26" role="img" aria-labelledby="title">
  <title id="title">cryoFILTER</title>
  <!-- App mark from index.html:17 and styles.css:72-84 at commit 081007c.
       IBM Plex Mono Medium glyph outlines; see fonts/OFL.txt and BRANDING.md. -->
  <rect x="0.25" y="0.25" width="25.5" height="25.5" rx="8" fill="#16222c" stroke="#2a3b49" stroke-width="0.5"/>
  <g fill="#4fb8d8" transform="translate({x:g} {y:g}) scale({scale:g} {-scale:g})">
    {chr(10).join(paths)}
  </g>
</svg>
'''
    (ASSETS / "logo.svg").write_text(svg)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        page = await browser.new_page(viewport={"width": 64, "height": 64}, device_scale_factor=1)
        await page.set_content(f'<style>html,body{{margin:0;background:transparent}}svg{{width:64px;height:64px;display:block}}</style>{svg}')
        await page.screenshot(path=str(ASSETS / "favicon.png"), omit_background=True)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())

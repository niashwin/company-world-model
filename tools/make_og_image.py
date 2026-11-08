"""Render website/assets/og.png, the image that link previews show (1200 x 630 pixels).

It needs Playwright with Chrome (pip install playwright; playwright install chrome) and the site on a local server:
    python -m http.server -d website 8765 &
    python tools/make_og_image.py
"""
import asyncio
import base64
import os
import re
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from playwright.async_api import async_playwright
SITE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/"
CARD = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
@font-face { font-family: "Geist"; font-weight: 100 900; src: url(data:font/woff2;base64,GEIST) format("woff2"); }
html, body { margin: 0; width: 1200px; height: 630px; overflow: hidden;
  background: repeating-linear-gradient(90deg, rgba(255,255,255,0) 0 54px, rgba(255,255,255,0.045) 54px 56px),
              linear-gradient(180deg, #041542 0%, #08205e 46%, #1a4aa3 84%, #2f6bd1 100%); }
.card { position: relative; width: 1200px; height: 630px; box-sizing: border-box; padding: 46px 64px 0; color: #fff; }
.logo { display: flex; align-items: center; gap: 8px; color: #fff; }
.logo svg { display: block; height: 26px; width: auto; }
.logo svg.word { height: 20px; }
h1 { font-family: "Geist"; font-weight: 500; font-size: 66px; line-height: 1.02; letter-spacing: -0.035em; margin: 30px 0 0; max-width: 15ch; }
p { font-family: "Geist"; font-size: 24px; color: rgba(255,255,255,0.82); margin: 16px 0 0; }
.fig { position: absolute; left: 64px; right: 64px; bottom: 0; height: 268px; overflow: hidden;
       border-radius: 12px 12px 0 0; background: #fff; box-shadow: 0 18px 48px -18px rgba(4,21,66,0.5); }
.fig img { width: 100%; display: block; }
</style></head><body><div class="card">
<div class="logo">MARK WORD</div>
<h1>Every system sees the same company</h1>
<p>Toward world models of organizations. Research from Sentra, 2026.</p>
<div class="fig"><img src="data:image/png;base64,IMG"></div>
</div></body></html>"""


def logo(name, cls):
    svg = open(os.path.join(ROOT, "website", "assets", "brand", name)).read()
    return re.sub(r"<svg ", f'<svg class="{cls}" ', svg, count=1)


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome", args=["--no-proxy-server"])
        ctx = await b.new_context(viewport={"width": 1136, "height": 900}, device_scale_factor=2, color_scheme="light")
        pg = await ctx.new_page()
        await pg.goto(SITE, wait_until="networkidle"); await pg.wait_for_timeout(1200)
        el = await pg.query_selector(".lanes-wrap")
        png = await el.screenshot()
        ctx2 = await b.new_context(viewport={"width": 1200, "height": 630}, device_scale_factor=1)
        pg2 = await ctx2.new_page()
        F = os.path.join(ROOT, "website", "assets", "fonts") + os.sep
        card = CARD.replace("GEIST", base64.b64encode(open(F + "geist-variable.woff2", "rb").read()).decode())
        card = card.replace("MARK", logo("sentra-logomark.svg", "mark")).replace("WORD", logo("sentra-wordmark.svg", "word"))
        await pg2.set_content(card.replace("IMG", base64.b64encode(png).decode()), wait_until="networkidle")
        await pg2.evaluate("document.fonts.ready")
        await pg2.wait_for_timeout(600)
        await pg2.screenshot(path=os.path.join(ROOT, "website", "assets", "og.png"))
        await b.close()
if __name__ == "__main__":
    asyncio.run(main())

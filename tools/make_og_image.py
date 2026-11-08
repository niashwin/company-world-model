"""Render website/assets/og.png, the image that link previews show (1200 x 630 pixels).

It needs Playwright with Chrome (pip install playwright; playwright install chrome) and the site on a local server:
    python -m http.server -d website 8765 &
    python tools/make_og_image.py
"""
import asyncio
import base64
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from playwright.async_api import async_playwright
SITE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/"
CARD = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
@font-face { font-family: "Archivo"; font-weight: 100 900; font-stretch: 62% 125%; src: url(data:font/woff2;base64,ARCHIVO) format("woff2"); }
@font-face { font-family: "Atkinson Hyperlegible Next"; font-weight: 200 800; src: url(data:font/woff2;base64,ATKINSON) format("woff2"); }
html, body { margin: 0; width: 1200px; height: 630px; background: #fbfcfd; overflow: hidden; }
.card { position: relative; width: 1200px; height: 630px; box-sizing: border-box; padding: 54px 64px 0; }
h1 { font-family: "Archivo"; font-weight: 700; font-stretch: 125%; font-size: 70px; line-height: 0.98; letter-spacing: -0.02em;
     color: #17212c; margin: 0; max-width: 15ch; }
p { font-family: "Atkinson Hyperlegible Next"; font-size: 25px; color: #4b5866; margin: 20px 0 0; }
.fig { position: absolute; left: 64px; right: 64px; bottom: 0; height: 318px; overflow: hidden;
       border: 1px solid #d5dce4; border-bottom: 0; border-radius: 12px 12px 0 0; background: #fbfcfd; }
.fig img { width: 100%; display: block; }
</style></head><body><div class="card">
<h1>Every system sees the same company</h1>
<p>Toward world models of organizations. Ashwin Gopinath, Sentra, 2026.</p>
<div class="fig"><img src="data:image/png;base64,IMG"></div>
</div></body></html>"""
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
        card = CARD.replace("ARCHIVO", base64.b64encode(open(F + "archivo-latin-wdth-normal.woff2", "rb").read()).decode())
        card = card.replace("ATKINSON", base64.b64encode(open(F + "atkinson-hyperlegible-next-latin-wght-normal.woff2", "rb").read()).decode())
        await pg2.set_content(card.replace("IMG", base64.b64encode(png).decode()), wait_until="networkidle")
        await pg2.evaluate("document.fonts.ready")
        await pg2.wait_for_timeout(600)
        await pg2.screenshot(path=os.path.join(ROOT, "website", "assets", "og.png"))
        await b.close()
if __name__ == "__main__":
    asyncio.run(main())

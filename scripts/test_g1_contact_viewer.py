#!/usr/bin/env python3
"""Browser smoke test; requires Playwright and system Chrome, not a GPU."""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://127.0.0.1:8089")
parser.add_argument("--screenshot", default="/tmp/g1-contact-viewer.png")
args = parser.parse_args()

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    page = browser.new_page(viewport={"width": 1600, "height": 1100})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("console", lambda message: print("CONSOLE", message.type, message.text, flush=True))
    page.on("requestfailed", lambda request: print("REQUEST_FAILED", request.url, request.failure, flush=True))
    page.goto(args.url, wait_until="domcontentloaded", timeout=30000)
    try:
        page.get_by_text("G1 + Revo2 contact inspection", exact=True).wait_for(timeout=30000)
    except Exception:
        print("BROWSER_FAILURE", page.url, page.locator("body").inner_text(), errors, flush=True)
        page.screenshot(path=args.screenshot)
        raise
    page.get_by_text("Focus hand", exact=True).click()
    page.wait_for_timeout(2500)
    print("BROWSER_DOM " + json.dumps(page.locator("body").inner_text()), flush=True)
    print("BROWSER_INPUTS " + json.dumps(page.locator("input").evaluate_all(
        "els => els.map(e => ({type:e.type, value:e.value, placeholder:e.placeholder}))")), flush=True)
    assert page.locator("canvas").count() > 0
    page.screenshot(path=args.screenshot)
    if errors:
        raise AssertionError(errors)
    print("BROWSER_PASS screenshot=" + str(Path(args.screenshot)), flush=True)
    browser.close()

#!/usr/bin/env python3
"""Browser smoke test; requires Playwright and system Chrome, not a GPU."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://127.0.0.1:8089")
parser.add_argument("--screenshot", default="/tmp/g1-contact-viewer.png")
args = parser.parse_args()

# Isaac startup is asynchronous; a refused connection is not a frontend bug.
for attempt in range(40):
    try:
        with urllib.request.urlopen(args.url, timeout=.5):
            break
    except OSError:
        if attempt == 39:
            raise
        time.sleep(.5)

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
    # A live dashboard must update and pause, not just display an HTTP page.
    state = page.locator('input[value^="sim "]')
    state.wait_for()
    running = page.locator('input[type="checkbox"]:visible').first
    running.uncheck(force=True)
    page.wait_for_timeout(500)
    paused = state.input_value()
    page.wait_for_timeout(1000)
    assert state.input_value() == paused, "pause does not stop physics"
    running.check(force=True)
    page.wait_for_function("old => document.querySelector('input[value^=\"sim \"]').value !== old", arg=paused)
    print("BROWSER_PASS pause/resume", flush=True)

    current_mode = "Pad probe"
    for target_mode in ("Back-of-finger probe", "Tool contact fixture", "Table contact fixture", "Free physics", "Pad probe"):
        # Mantine omits the type attribute on its visible select input; the
        # HTML property is "text", but a [type="text"] selector won't match.
        page.locator(f'input[value="{current_mode}"]:visible').click()
        page.get_by_role("option", name=target_mode, exact=True).click()
        page.wait_for_function("mode => document.querySelector('input[value^=\"sim \"]').value.endsWith(mode)", arg=target_mode)
        page.wait_for_timeout(600)
        print("BROWSER_PASS mode=" + target_mode, flush=True)
        current_mode = target_mode
    page.get_by_text("G1 + Revo2 contact inspection", exact=True).scroll_into_view_if_needed()
    print("BROWSER_READOUTS " + json.dumps(page.locator("input:disabled").evaluate_all("els => els.map(e => e.value)")), flush=True)
    assert page.locator("canvas").count() > 0
    page.screenshot(path=args.screenshot)
    if errors:
        raise AssertionError(errors)
    print("BROWSER_PASS screenshot=" + str(Path(args.screenshot)), flush=True)
    browser.close()

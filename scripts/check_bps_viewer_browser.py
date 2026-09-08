#!/usr/bin/env python3
"""Exercise the live BPS viewer controls and capture browser validation evidence."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright, expect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Existing viewer artifact directory")
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()
    with np.load(args.output / "bps_128.npz") as data:
        distances = data["distances"]
    errors, checks = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=[
            "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(f"http://127.0.0.1:{args.port}", wait_until="domcontentloaded")
        page.get_by_text("Object shape | BPS-128", exact=True).wait_for(timeout=30000)
        page.wait_for_timeout(2000)
        slider = page.get_by_role("slider")
        expect(slider).to_have_attribute("aria-valuenow", "16")
        expect(page.get_by_text(f"BPS[16] = {distances[16]:.4f}", exact=True)).to_be_visible()
        page.screenshot(path=str(args.output / "overview.png"))
        checks.append("Initial index and displayed value match the saved descriptor")

        slider.focus()
        slider.press("Home")
        expect(slider).to_have_attribute("aria-valuenow", "0")
        expect(page.get_by_text(f"BPS[0] = {distances[0]:.4f}", exact=True)).to_be_visible()
        slider.press("End")
        expect(slider).to_have_attribute("aria-valuenow", "127")
        expect(page.get_by_text(f"BPS[127] = {distances[127]:.4f}", exact=True)).to_be_visible()
        checks.append("Both index extremes update the descriptor value")
        slider.press("Home")
        for _ in range(16):
            slider.press("ArrowRight")
        expect(page.get_by_text(f"BPS[16] = {distances[16]:.4f}", exact=True)).to_be_visible()
        page.get_by_role("checkbox", name="Only selected connection", exact=True).check()
        page.wait_for_timeout(500)
        page.screenshot(path=str(args.output / "selected_connection.png"))
        page.get_by_role("checkbox", name="16,384 surface samples", exact=True).check()
        page.get_by_role("checkbox", name="Object mesh", exact=True).uncheck()
        page.wait_for_timeout(500)
        page.screenshot(path=str(args.output / "surface_samples.png"))
        checks.append("Selected-only and surface-cloud views render")
        # Leave the user's scene in the clean default state, even across clients.
        page.get_by_role("checkbox", name="Object mesh", exact=True).check()
        page.get_by_role("checkbox", name="16,384 surface samples", exact=True).uncheck()
        page.get_by_role("checkbox", name="Only selected connection", exact=True).uncheck()
        page.get_by_role("button", name="Reset camera", exact=True).click()
        page.wait_for_timeout(500)
        text = page.locator("body").inner_text()
        browser.close()

    # Catch blank WebGL scenes and verify that selected-only really hides the
    # amber query markers, not just that an HTML checkbox changed state.
    amber_counts = {}
    for name in ("overview", "selected_connection"):
        pixels = np.asarray(Image.open(args.output / f"{name}.png").convert("RGB"))[:, :1100].astype(int)
        amber = ((pixels[..., 0] > 180) & (pixels[..., 1] > 100)
                 & (pixels[..., 1] < 225) & (pixels[..., 2] < 110))
        amber_counts[name] = int(amber.sum())
    assert amber_counts["overview"] > 1000, amber_counts
    assert amber_counts["selected_connection"] < amber_counts["overview"] * .2, amber_counts
    checks.append("Rendered amber pixels disappear in selected-only mode")
    report = {"browser_errors": errors, "checks": checks,
              "amber_pixel_counts": amber_counts, "final_ui_text": text}
    (args.output / "browser_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert not errors, errors


if __name__ == "__main__":
    main()

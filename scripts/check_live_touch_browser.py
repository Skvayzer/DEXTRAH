#!/usr/bin/env python3
"""Browser-level check of the actual live PhysX viewer; no simulated UI data."""
import argparse
import json
import re
import time
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8091)
    parser.add_argument('--output',type=Path,default=Path('outputs/revo2_touch_live'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        page=browser.new_page(viewport={'width':1550,'height':1050})
        page.on('pageerror',lambda e: errors.append(str(e)))
        page.goto(f'http://127.0.0.1:{args.port}',wait_until='domcontentloaded')
        page.get_by_text('Live Revo2 contact physics',exact=True).wait_for(timeout=30000)
        pause=page.get_by_role('checkbox',name='Run physics',exact=True)
        state=page.get_by_text('physics time',exact=False)
        def capture_loaded(name):
            page.get_by_role('button',name='Run selected test',exact=True).click()
            expect(pause).to_be_checked()
            deadline=time.monotonic()+45
            while time.monotonic()<deadline:
                match=re.search(r'test time ([\d.]+)',state.inner_text())
                assert 'SAFETY STOP' not in state.inner_text()
                if match and 2.2 <= float(match[1]) < 4.:
                    pause.uncheck()
                    break
                page.wait_for_timeout(60)
            else:
                raise AssertionError('Live test did not reach its loaded phase')
            page.wait_for_timeout(400)
            # Actual force table, not merely successful UI button handling.
            row=page.get_by_role('row').filter(has=page.get_by_role('cell',name='index',exact=True))
            assert float(row.get_by_role('cell').nth(1).inner_text())>.03
            page.screenshot(path=str(args.output/name))
        capture_loaded('live_press.png')
        before=state.inner_text()
        page.wait_for_timeout(600)
        assert state.inner_text()==before, 'pause did not hold simulation'
        # All data here come from the running simulation. Start a shear test.
        page.get_by_role('combobox').nth(1).click()
        page.get_by_role('option',name='Static shear',exact=True).click()
        capture_loaded('live_shear.png')
        page.get_by_role('button',name='Release / cancel',exact=True).click()
        page.wait_for_timeout(800)
        expect(state).to_contain_text('Idle')
        page.screenshot(path=str(args.output/'live_release.png'))
        page.get_by_role('button',name='Reset camera',exact=True).click()
        browser.close()
    report={'browser_errors':errors,'checks':['live page loaded','pause holds physics time',
        'loaded press and shear show nonzero simulated fingertip force',
        'static shear command resumes physics','release returns to idle'],
        'force_source':'running PhysX bench, not recording replay'}
    (args.output/'live_browser_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert not errors


if __name__=='__main__':
    main()

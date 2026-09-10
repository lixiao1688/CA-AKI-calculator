# -*- coding: utf-8 -*-
"""keepalive.py — 定时唤醒 Streamlit Community Cloud 应用。
HTTP 请求无法唤醒（返回静态 shell，不启动 Python 进程）；必须用无头浏览器
建立 WebSocket 连接。本脚本用 Playwright 每 ~6 小时访问一次，保持应用清醒。
由 GitHub Actions 定时调度（.github/workflows/keepalive.yml）。
"""
from playwright.sync_api import sync_playwright

URL = "https://ca-aki-calculator.streamlit.app/"


def keepalive():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(URL, wait_until="domcontentloaded", timeout=60000)
            # 若出现睡眠页，点击唤醒按钮
            try:
                wake = page.get_by_role("button", name="Yes, get this app back up!")
                if wake.count() > 0:
                    print("WAKE: app sleeping, clicking wake-up button")
                    wake.click()
            except Exception:
                pass
            # 给冷启动足够时间，等待应用真正渲染（WebSocket 连接建立）
            page.wait_for_timeout(90000)
            if page.locator("text=CA-AKI").count() > 0:
                print("OK: app is awake and running")
            else:
                print("WARN: app content not confirmed")
        except Exception as e:
            print(f"ERROR: {e}")
        finally:
            browser.close()


if __name__ == "__main__":
    keepalive()

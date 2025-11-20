# -*- coding: utf-8 -*-
"""scraper.py (regular‑Chrome version)

This version swaps **undetected‑chromedriver** for Selenium’s standard
`webdriver.Chrome`, while retaining all other logic and the configurable
`HEADLESS` flag. If you need stealth features again, re‑add
undetected‑chromedriver, but for now the script relies only on the Selenium
package (≥v4.18, which bundles selenium‑manager to fetch the right driver).

Run requirements:
    pip install selenium

Notes
-----
* `webdriver.Chrome()` automatically downloads the correct ChromeDriver binary
  via selenium‑manager when Chrome ≥115 is installed. If your environment needs
  a custom driver path, adjust the `Service` object accordingly.
* A minimal anti‑bot measure remains: the script disables the
  `AutomationControlled` Blink feature.
"""
print("Importing Dependencies")
from __future__ import annotations

import csv
import logging
import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Set

# Selenium imports (regular Chrome)
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# ---------- Config & helpers ---------- #
LINK_PATH = "data/tweets/link/"
RAW_TWEET_PATH = "data/tweets/raw/"

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s ‑ %(message)s",
    datefmt="%Y‑%m‑%d %H:%M:%S",
)

HEADLESS = False  # Set to True to run without a visible window

SCROLL_PAUSE_RANGE = (5, 10)  # (min, max) seconds for human‑ish delays
SCROLLS_BEFORE_PAUSE = 8          # Longer pause every N scrolls
LONG_PAUSE_SEC = 6
BACKOFF_BASE = 15                 # Base seconds for exponential back‑off


def human_pause(min_s: float = SCROLL_PAUSE_RANGE[0], max_s: float = SCROLL_PAUSE_RANGE[1]) -> None:
    """Sleep for a random time between *min_s* and *max_s* seconds."""
    time.sleep(random.uniform(min_s, max_s))

# ---------- Core scraping helpers ---------- #

def collect_links(
    driver: webdriver.Chrome,
    username: str,
    links_path: str | Path,
    max_scrolls: int = 300,
) -> None:
    """Scroll through *username*'s timeline and save unique tweet permalinks to CSV.

    Output file: `{links_path}/{username}_links.csv` (single column `link`).
    """
    links_path = Path(links_path)
    links_path.mkdir(parents=True, exist_ok=True)
    file_path = links_path / f"{username}_links.csv"

    # Read existing links to avoid duplicates
    seen_links: Set[str] = set()
    if file_path.exists():
        with file_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            seen_links.update(row["link"] for row in reader)
            logging.info("Loaded %d existing links for @%s", len(seen_links), username)

    driver.get(f"https://x.com/{username}")

    wait = WebDriverWait(driver, 10)
    try:
        wait.until(EC.presence_of_element_located((By.XPATH, "//article")))
    except TimeoutException:
        logging.warning("Timeout waiting for tweets on @%s page. Skipping.", username)
        return

    tweet_links: List[str] = []
    last_height = driver.execute_script("return document.body.scrollHeight")

    for i in range(max_scrolls):
        anchors = driver.find_elements(By.XPATH, "//a[contains(@href, '/status/') and descendant::time]")
        for a in anchors:
            link = a.get_attribute("href")
            if link and link not in seen_links:
                tweet_links.append(link)
                seen_links.add(link)

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        human_pause()

        if (i + 1) % SCROLLS_BEFORE_PAUSE == 0:
            time.sleep(LONG_PAUSE_SEC)

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            logging.info("No further scroll growth after %d iterations on @%s", i + 1, username)
            break
        last_height = new_height

    if not tweet_links:
        logging.info("Found no new links for @%s", username)
        return

    write_header = not file_path.exists() or file_path.stat().st_size == 0
    with file_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["link"])
        writer.writerows([[l] for l in tweet_links])
    logging.info("Saved %d new links for @%s", len(tweet_links), username)


def scrape_tweets(
    driver: webdriver.Chrome,
    username: str,
    links_path: str | Path,
    out_path: str | Path,
    days_back: int = 365,
) -> None:
    """Visit saved tweet links, scrape tweet text + timestamp, and append to CSV.

    Output file: `{out_path}/{username}_tweets.csv` with columns [`link`, `time`, `tweet`].
    Tweets older than *days_back* days are skipped.
    """
    links_path = Path(links_path)
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    links_file = links_path / f"{username}_links.csv"
    if not links_file.exists():
        logging.warning("Links file not found for @%s at %s. Run collect_links first.", username, links_file)
        return

    tweets_file = out_path / f"{username}_tweets.csv"

    with links_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_links = [row["link"] for row in reader]

    scraped_links: Set[str] = set()
    if tweets_file.exists():
        with tweets_file.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            scraped_links.update(row["link"] for row in reader)

    threshold_date = datetime.now(timezone.utc) - timedelta(days=days_back)

    write_header = not tweets_file.exists() or tweets_file.stat().st_size == 0
    with tweets_file.open("a", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        if write_header:
            writer.writerow(["link", "time", "tweet"])

        for idx, link in enumerate(all_links, 1):
            if link in scraped_links:
                continue

            retries = 0
            while retries < 3:
                try:
                    driver.get(link)
                    wait = WebDriverWait(driver, 10)
                    time_el = wait.until(EC.presence_of_element_located((By.XPATH, "//a/time")))
                    tweet_time = datetime.fromisoformat(time_el.get_attribute("datetime").replace("Z", "+00:00"))

                    article = wait.until(EC.presence_of_element_located((By.XPATH, "//article")))
                    tweet_text = article.text.replace("\n", " ")
                    writer.writerow([link, tweet_time.isoformat(), tweet_text])
                    scraped_links.add(link)
                    logging.info("Scraped %d/%d tweet links for @%s", idx, len(all_links), username)
                    human_pause()
                    break
                except TimeoutException as e:
                    retries += 1
                    backoff = BACKOFF_BASE * (2 ** (retries - 1))
                    logging.warning("Timeout (%s) on link %s – retry %d/3 after %ds", e.__class__.__name__, link, retries, backoff)
                    time.sleep(backoff)
                except WebDriverException as e:
                    logging.error("WebDriver error on %s: %s", link, e)
                    break

# ---------- Entry‑point ---------- #

def build_driver() -> webdriver.Chrome:
    """Create and return a configured Selenium Chrome driver."""
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    if HEADLESS:
        # Chrome ≥109 supports the new headless mode flag.
        options.add_argument("--headless=new")
    return webdriver.Chrome(options=options)


if __name__ == "__main__":
    usernames = [
        "ecoyuri",
        "shinji_ishimaru",
        "renho_sha",
        # "toshio_tamogami",
    ]

    driver = build_driver()
    driver.get("https://x.com/i/flow/login")
    input("Log in to X/Twitter in the opened window, then press <Enter> here to continue…")

    try:
        for user in usernames:
            collect_links(driver, user, links_path=LINK_PATH)

        for user in usernames:
            scrape_tweets(driver, user, links_path=LINK_PATH, out_path=RAW_TWEET_PATH)

    finally:
        try:
            driver.quit()
        except Exception:
            pass
        logging.info("Finished. Browser closed.")

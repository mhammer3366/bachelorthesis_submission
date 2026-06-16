import os
import csv
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin
import re
import logging
from playwright.sync_api import sync_playwright, Page 

# --- Configuration ---
BASE_URL = "https://www.srf.ch"
AZ_URL = f"{BASE_URL}/audio/a-z"
OUTPUT_BASE_DIR = "srf_audio_downloads"
METADATA_FILE = "srf_audio_metadata.csv"

# Ensure the base output directory exists
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# HEADERS for direct requests (like A-Z page and audio file downloads)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("srf_scraper.log"),
        logging.StreamHandler()
    ]
)

# --- Helper Functions ---

def sanitize_filename(name):
    """Sanitizes a string to be used as a filename or directory name."""
    # Remove invalid characters: / \ : * ? " < > |
    sanitized_name = re.sub(r'[\\/:*?"<>|]', '', name)
    # Replace spaces with underscores, or keep as is. Trim whitespace.
    sanitized_name = sanitized_name.strip()
    return sanitized_name

def get_series_links():
    """
    Fetches all unique links to podcast series from the A-Z index page.
    This still uses requests because the A-Z page usually loads without JS.
    """
    logging.info(f"🔎 Loading A-Z page: {AZ_URL}")
    try:
        resp = requests.get(AZ_URL, headers=HEADERS, timeout=10)
        resp.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Failed to access A-Z page {AZ_URL}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    links = []

    # Updated selector based on recent SRF A-Z structure observations
    # Look for a.layered-link or a within an li.c-column-list__item with href starting /audio/
    for a_tag in soup.select("li.c-column-list__item a[href^='/audio/'], a.layered-link[href^='/audio/']"):
        href = a_tag.get("href")
        if href and "/audio/podcast/" in href: # Prefer podcast-specific links if available
            full_url = urljoin(BASE_URL, href)
            if full_url not in links:
                links.append(full_url)
        elif href and "/audio/" in href: # General audio links
            full_url = urljoin(BASE_URL, href)
            if full_url not in links:
                links.append(full_url)

    if not links:
        logging.warning("⚠️ No series links found. The site structure may have changed.")
    else:
        logging.info(f"✅ Found {len(links)} series links.")
    return links

def get_series_info(series_url: str, page_browser: Page):
    """
    Fetches the series name and episode links for a given series URL, handling pagination,
    using Playwright for full page rendering.
    """
    episode_data = []
    series_name = "Unknown Podcast Series"
    page_iteration = 0

    logging.info(f"\n➡️ Processing series URL: {series_url}")

    try:
        # Navigate to the series URL
        page_browser.goto(series_url, wait_until="domcontentloaded", timeout=30000) # Increased timeout

        # Wait for potential dynamic content (e.g., episode list) to load
        try:
            page_browser.wait_for_selector(
                "li.radio-episode, a.audio-teaser__title-link, .mod-item, .teaser-item, .article-teaser",
                state="visible", # Ensure it's visible on the page
                timeout=15000 # Wait up to 15 seconds for initial episode elements
            )
            logging.debug("  DEBUG: Found episode-like elements after initial wait.")
        except Exception:
            logging.warning("  WARNING: No direct episode list elements visible after initial wait. Proceeding with current HTML.")

        # Extract series name (from the initially loaded page)
        initial_html_content = page_browser.content()
        initial_soup = BeautifulSoup(initial_html_content, "html.parser")

        series_title_tag = initial_soup.find("meta", property="og:title")
        if series_title_tag:
            series_name = series_title_tag["content"].strip()
        else:
            alt_title = initial_soup.select_one(".mod-header__title, h1.mod-header__title, h2.radio-show-detail__title")
            if alt_title:
                series_name = alt_title.text.strip()
            elif "audio/" in series_url:
                parts = series_url.split('/audio/')
                if len(parts) > 1:
                    name_from_url = parts[1].split('/')[0].replace('-', ' ').title()
                    if name_from_url:
                        series_name = name_from_url

        series_name = sanitize_filename(series_name)
        logging.info(f"  Identified series name: '{series_name}'")

        # Helper to extract all episode URLs from a given BeautifulSoup object
        def extract_episode_urls_from_soup(soup_obj):
            extracted_urls = set()
            # 1. 'radio-episode'
            for item in soup_obj.select("li.radio-episode"):
                episode_link_tag = item.select_one("a.radio-episode__link")
                direct_download_link_tag = item.select_one(".radio-episode__download a")
                if episode_link_tag:
                    extracted_urls.add(urljoin(BASE_URL, episode_link_tag.get("href")))
                if direct_download_link_tag:
                    extracted_urls.add(direct_download_link_tag.get("href"))
            # 2. 'audio-teaser__title-link'
            for link in soup_obj.select("a.audio-teaser__title-link"):
                href = link.get("href")
                if href:
                    full_episode_url = urljoin(BASE_URL, href)
                    extracted_urls.add(full_episode_url)
            # 3. Generic fallback for any link that might lead to an audio page
            for container in soup_obj.select(".mod-item, .teaser-item, .article-teaser"):
                link_in_container = container.select_one("a[href*='/audio/']")
                if link_in_container:
                    href = link_in_container.get("href")
                    if href and not href.endswith(".mp3"): # Exclude direct MP3 links here if they are not episode pages
                        full_episode_url = urljoin(BASE_URL, href)
                        extracted_urls.add(full_episode_url)
            return extracted_urls

        # --- Pagination loop with Playwright ---
        while True:
            current_html_content = page_browser.content()
            soup_before_click = BeautifulSoup(current_html_content, "html.parser")

            # Extract episode URLs from the current page content *before* clicking "Load More"
            episodes_on_page_before_click = extract_episode_urls_from_soup(soup_before_click)
            
            # Add these to our main episode_data list (deduplication will happen at the end)
            # We iterate through the set to ensure unique additions
            for url in episodes_on_page_before_click:
                # Add a basic dict; direct_audio_url can be None and fetched later if needed
                # Only add if it's not already in episode_data to avoid redundant objects for deduplication later
                if {"episode_url": url, "direct_audio_url": None} not in episode_data:
                     # Check for existing dict with this URL, then add.
                     # A more robust approach involves a set of (episode_url, direct_audio_url) tuples for `episode_data` itself
                     # but for now, this append and final deduplication is okay.
                    episode_data.append({"episode_url": url, "direct_audio_url": None}) 


            logging.info(f"  📄 Page iteration {page_iteration}: Found {len(episodes_on_page_before_click)} unique episode URLs on this page.")
            
            # --- Pagination button logic ---
            load_more_button_selector = "button.button--load-more, a.button--load-more, .srf-button--load-more, button[aria-label='Mehr laden'], .js-load-more-button, .js-show-more-button"
            
            next_button = page_browser.query_selector(load_more_button_selector)

            if next_button and next_button.is_visible() and not next_button.is_disabled():
                logging.info(f"  Clicking 'Load More' button for '{series_name}' (iteration {page_iteration})...")
                next_button.click()
                
                # Wait for network idle state or specific elements to appear
                page_browser.wait_for_load_state("networkidle", timeout=60000) # Increased timeout to 60 seconds
                time.sleep(2) # Increased sleep to 2 seconds

                # Re-parse the HTML after the click and wait
                updated_html_content = page_browser.content()
                soup_after_click = BeautifulSoup(updated_html_content, "html.parser")
                
                # Extract episode URLs again from the page content *after* clicking "Load More"
                episodes_on_page_after_click = extract_episode_urls_from_soup(soup_after_click)
                
                # Crucial check: Did the number of unique episode URLs on the page actually increase?
                if len(episodes_on_page_after_click) <= len(episodes_on_page_before_click):
                    logging.info(f"  No new unique episode URLs found on the page after clicking 'Load More'. "
                                 f"Before: {len(episodes_on_page_before_click)}, After: {len(episodes_on_page_after_click)}. "
                                 f"Assuming end of dynamic loading or a different pagination method.")
                    break # Break if no new elements appeared or count didn't increase
                
                page_iteration += 1 # Increment for the next iteration
            else:
                logging.info("  No 'Load More' button found or it's disabled. Ending pagination for this series.")
                break

    except Exception as e:
        logging.error(f"❌ Error getting series info for {series_url}: {e}")
        # Optional: Save a screenshot on error for debugging
        try:
            screenshot_path = f"error_series_{sanitize_filename(series_name)}.png"
            # Note: Screenshots only work if headless=False AND an X server is available.
            # If running on a headless server, this will fail but won't stop the script.
            page_browser.screenshot(path=screenshot_path)
            logging.error(f"  Screenshot saved to {screenshot_path}")
        except Exception as scr_e:
            logging.error(f"  Failed to save screenshot: {scr_e}")

    # Deduplicate episode_data based on episode_url or direct_audio_url at the very end
    unique_episode_data = []
    seen_keys = set()
    for item in episode_data:
        key = item.get('episode_url') or item.get('direct_audio_url')
        if key and key not in seen_keys:
            unique_episode_data.append(item)
            seen_keys.add(key)
        elif key:
            logging.debug(f"  DEBUG: Skipping duplicate episode in final deduplication: {key}")

    logging.info(f"  Total unique episodes collected for '{series_name}': {len(unique_episode_data)}")
    return series_name, unique_episode_data

def get_audio_info(episode_urls_data: dict, podcast_series_name: str) -> dict or None:
    """
    Extracts detailed audio information (title, audio URL) from an episode page.
    Prioritizes direct_audio_url if available from the series page.
    """
    episode_url = episode_urls_data.get("episode_url")
    direct_audio_url_from_list = episode_urls_data.get("direct_audio_url")

    # If a direct audio URL was found on the series list page, use it directly
    if direct_audio_url_from_list:
        title = "Untitled Episode"
        # Try to infer title from episode_url slug if available
        if episode_url:
            match = re.search(r'/audio/[^/]+/(?P<episode_slug>[^/?]+)', episode_url)
            if match:
                title = match.group('episode_slug').replace('-', ' ').title()
        
        # Fallback for title if not found from episode_url
        if title == "Untitled Episode":
             title_from_audio_basename = os.path.basename(direct_audio_url_from_list.split('?')[0]).replace('-', ' ').title()
             if title_from_audio_basename and not title_from_audio_basename.startswith("Aud"): # Avoid generic "Audio" titles
                 title = title_from_audio_basename

        filename = sanitize_filename(os.path.basename(direct_audio_url_from_list.split("?")[0]))
        if not filename.endswith(".mp3"): # Ensure it has a typical audio extension
            filename += ".mp3"

        logging.info(f"    🎧 Using direct audio URL for: '{title}' (from series list)")
        return {
            "title": title,
            "podcast_name": podcast_series_name,
            "url": episode_url if episode_url else direct_audio_url_from_list, # Use episode_url as main link if available
            "audio_url": direct_audio_url_from_list,
            "filename": filename
        }

    if not episode_url:
        logging.warning("    ⚠️ Neither episode_url nor direct_audio_url found for an item. Skipping.")
        return None

    logging.info(f"    🎧 Fetching episode details from: {episode_url}")
    try:
        resp = requests.get(episode_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        audio_url = None
        # Priority 1: og:audio meta tag
        audio_url_meta = soup.find("meta", property="og:audio")
        if audio_url_meta and audio_url_meta.get("content"):
            audio_url = audio_url_meta.get("content")
        
        # Priority 2: direct download link button
        if not audio_url:
            download_button_link = soup.select_one(".radio-episode__download a")
            if download_button_link and download_button_link.get("href"):
                audio_url = download_button_link.get("href")

        # Priority 3: <audio> tag src
        if not audio_url:
            audio_tag = soup.find("audio")
            if audio_tag and audio_tag.get("src"):
                audio_url = audio_tag["src"]
        
        # Priority 4: <source> tag within <audio>
        if not audio_url:
            audio_source = soup.find("source", type="audio/mpeg")
            if audio_source and audio_source.get("src"):
                audio_url = audio_source.get("src")
        
        # Ensure audio_url is absolute
        if audio_url and not audio_url.startswith("http"):
            audio_url = urljoin(BASE_URL, audio_url)

        if not audio_url:
            logging.warning(f"    ⚠️ No valid audio source found for {episode_url}. Skipping.")
            return None

        # Extract title
        title_tag = soup.find("meta", property="og:title")
        title = sanitize_filename(title_tag["content"].strip()) if title_tag else "Untitled"
        if title == "Untitled": # Fallback if og:title not found or generic
            h1_tag = soup.select_one("h1.mod-header__title, h1.radio-episode__title")
            if h1_tag:
                title = sanitize_filename(h1_tag.text.strip())
            else: # Try to infer from URL if all else fails
                match = re.search(r'/audio/[^/]+/(?P<episode_slug>[^/?]+)', episode_url)
                if match:
                    title = match.group('episode_slug').replace('-', ' ').title()
                else:
                    title = os.path.basename(episode_url.split('?')[0]).replace('-', ' ').title()


        filename = os.path.basename(audio_url.split("?")[0])
        filename = sanitize_filename(filename)
        if not filename.endswith(".mp3"): # Add .mp3 extension if missing
            filename += ".mp3"

        return {
            "title": title,
            "podcast_name": podcast_series_name,
            "url": episode_url,
            "audio_url": audio_url,
            "filename": filename
        }
    except requests.exceptions.RequestException as e:
        logging.error(f"    ❌ Network error fetching episode {episode_url}: {e}")
        return None
    except Exception as e:
        logging.error(f"    ❌ Error parsing episode {episode_url}: {e}")
        return None

def download_audio(url: str, path: str):
    """Downloads an audio file to the specified path."""
    if os.path.exists(path):
        logging.info(f"    ⏩ File already exists, skipping download: {path}")
        return

    logging.info(f"    ⏳ Downloading: {os.path.basename(path)}")
    try:
        with requests.get(url, stream=True, headers=HEADERS, timeout=60) as r: # Increased timeout for large files
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 8192


            with open(path, "wb") as f:
                for chunk in tqdm(r.iter_content(block_size),
                                  total=(total_size + block_size - 1) // block_size if total_size else None,
                                  unit='KB', unit_scale=True,
                                  desc=f"      Downloading {os.path.basename(path)}", leave=False):
                    f.write(chunk)
        logging.info(f"    ✅ Downloaded: {path}")
    except requests.exceptions.RequestException as e:
        logging.error(f"    ❌ Failed to download {url} to {path}: {e}")
    except Exception as e:
        logging.error(f"    ❌ An unexpected error occurred during download {path}: {e}")

def save_metadata(data: list[dict], filepath: str):
    """Saves episode metadata to a CSV file."""
    fieldnames = ["filename", "title", "podcast_name", "url", "audio_url"]
    file_exists = os.path.exists(filepath)

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for row_data in data:
            row_to_write = {field: row_data.get(field, "") for field in fieldnames}
            writer.writerow(row_to_write)
    logging.info(f"Metadata saved/appended to {filepath}")

# --- Main Scraper Logic ---

def main():
    all_metadata_for_current_run = []

    logging.info("Starting SRF Audio Scraper...")
    series_links = get_series_links()

    if not series_links:
        logging.error("❌ No series links found. Exiting.")
        return

    # Initialize Playwright once to reuse the browser instance
    with sync_playwright() as p:
        # Set headless=True for running on servers without a GUI
        browser = p.chromium.launch(headless=True) 
        # Create a new page and set the User-Agent
        page = browser.new_page(user_agent=HEADERS['User-Agent'])

        for series_url in tqdm(series_links, desc="📚 Overall Series Progress"):
            time.sleep(2) # Delay between processing each series to be polite

            podcast_name, episode_list_data = get_series_info(series_url, page)
            
            # Final deduplication of episodes (in case some were added multiple times from different selectors)
            unique_episode_data = []
            seen_keys = set()
            for item in episode_list_data:
                # Use episode_url as primary key, fallback to direct_audio_url if episode_url is None
                key = item.get('episode_url') or item.get('direct_audio_url')
                if key and key not in seen_keys:
                    unique_episode_data.append(item)
                    seen_keys.add(key)
                elif key:
                    logging.debug(f"  DEBUG: Final deduplication skipped: {key}")

            if not unique_episode_data:
                logging.warning(f"  No unique episodes found for series: '{podcast_name}' ({series_url}) after Playwright processing.")
                continue
                
            podcast_output_dir = os.path.join(OUTPUT_BASE_DIR, podcast_name)
            os.makedirs(podcast_output_dir, exist_ok=True)

            for ep_data in tqdm(unique_episode_data, desc=f"  Episodes in '{podcast_name}'", leave=False):
                info = get_audio_info(ep_data, podcast_name)
                if not info:
                    continue # Skip if audio info couldn't be retrieved
                
                filepath = os.path.join(podcast_output_dir, info["filename"])
                
                download_audio(info["audio_url"], filepath)
                
                # Store relative path in metadata for portability
                info["filename"] = os.path.join(podcast_output_dir, info["filename"]) 
                all_metadata_for_current_run.append(info)
                time.sleep(0.5) # Small delay between episode downloads
        
        browser.close() # Close the browser when done with all series

    save_metadata(all_metadata_for_current_run, METADATA_FILE)
    logging.info(f"\n✅ Done. Attempted to process {len(all_metadata_for_current_run)} episodes.")
    logging.info(f"Check '{METADATA_FILE}' for metadata and '{OUTPUT_BASE_DIR}' for downloaded files.")

if __name__ == "__main__":
    main()
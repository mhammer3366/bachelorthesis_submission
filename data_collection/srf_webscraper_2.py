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
        # We still need series page URLs to navigate to them with Playwright
        if href and "/audio/" in href:
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
    Fetches the series name and direct MP3 links for a given series URL, handling pagination,
    using Playwright for full page rendering. Only collects direct MP3 links.
    """
    episode_data = [] # This will store {"episode_url": direct_audio_link, "direct_audio_url": None}
    series_name = "Unknown Podcast Series"
    page_iteration = 0

    logging.info(f"\n➡️ Processing series URL: {series_url}")

    try:
        # Navigate to the series URL
        page_browser.goto(series_url, wait_until="domcontentloaded", timeout=30000)

        # Wait for potential dynamic content (e.g., episode list) to load
        try:
            # We specifically wait for direct download links to appear or general containers that might hold them
            page_browser.wait_for_selector(
                ".radio-episode__download a[href$='.mp3'], .radio-episode__download a[href*='download-media.srf.ch'], a[href$='.mp3'], a[href*='download-media.srf.ch']",
                state="visible",
                timeout=15000
            )
            logging.debug("  DEBUG: Found direct audio link elements after initial wait.")
        except Exception:
            logging.warning("  WARNING: No direct audio link elements visible after initial wait. Proceeding with current HTML.")

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

        # Helper to extract ONLY direct MP3 URLs from a given BeautifulSoup object
        def extract_only_direct_mp3_urls_from_soup(soup_obj):
            extracted_urls = set() # Use a set for unique URLs
            
            # 1. 'radio-episode' - ONLY collect direct download links
            for item in soup_obj.select("li.radio-episode"):
                direct_download_link_tag = item.select_one(".radio-episode__download a")
                if direct_download_link_tag:
                    href = direct_download_link_tag.get("href")
                    if href and ".mp3" in href: # Ensure it's a direct MP3 link
                        extracted_urls.add(href)
                # Removed fallback to episode page link

            # 2. 'audio-teaser__title-link' - These typically lead to episode pages, so we skip them
            # Removed logic for this selector

            # 3. Generic fallback - STRICTLY ONLY DIRECT AUDIO OR DOWNLOAD DOMAIN
            for container in soup_obj.select(".mod-item, .teaser-item, .article-teaser"):
                # Look for direct MP3 links within these containers
                link_in_container = container.select_one("a[href$='.mp3']") # Links ending in .mp3
                if not link_in_container:
                    # Or links specifically from the SRF download domain
                    link_in_container = container.select_one("a[href*='download-media.srf.ch']")
                
                if link_in_container:
                    href = link_in_container.get("href")
                    if href:
                        # Ensure it's an absolute URL
                        full_audio_url = urljoin(BASE_URL, href)
                        extracted_urls.add(full_audio_url)
            
            return extracted_urls

        # --- Pagination loop with Playwright ---
        while True:
            current_html_content = page_browser.content()
            soup_before_click = BeautifulSoup(current_html_content, "html.parser")

            # Extract ONLY direct MP3 URLs from the current page content *before* clicking "Load More"
            episodes_on_page_before_click = extract_only_direct_mp3_urls_from_soup(soup_before_click)
            
            # Add these to our main episode_data list
            for url in episodes_on_page_before_click:
                # Store the direct MP3 URL as 'episode_url'
                if {"episode_url": url, "direct_audio_url": None} not in episode_data:
                    episode_data.append({"episode_url": url, "direct_audio_url": None}) 

            logging.info(f"  📄 Page iteration {page_iteration}: Found {len(episodes_on_page_before_click)} unique direct MP3 URLs on this page.")
            
            # --- Pagination button logic ---
            load_more_button_selector = "button.button--load-more, a.button--load-more, .srf-button--load-more, button[aria-label='Mehr laden'], .js-load-more-button, .js-show-more-button"
            
            next_button = page_browser.query_selector(load_more_button_selector)

            if next_button and next_button.is_visible() and not next_button.is_disabled():
                logging.info(f"  Clicking 'Load More' button for '{series_name}' (iteration {page_iteration})...")
                next_button.click()
                
                # Wait for network idle state or specific elements to appear
                page_browser.wait_for_load_state("networkidle", timeout=60000) # Keep max timeout for robustness
                time.sleep(1) # Reduced fixed sleep time
                
                # Re-parse the HTML after the click and wait
                updated_html_content = page_browser.content()
                soup_after_click = BeautifulSoup(updated_html_content, "html.parser")
                
                # Extract episode URLs again from the page content *after* clicking "Load More"
                episodes_on_page_after_click = extract_only_direct_mp3_urls_from_soup(soup_after_click)
                
                # Crucial check: Did the number of unique episode URLs on the page actually increase?
                if len(episodes_on_page_after_click) <= len(episodes_on_page_before_click):
                    logging.info(f"  No new unique direct MP3 URLs found on the page after clicking 'Load More'. "
                                 f"Before: {len(episodes_on_page_before_click)}, After: {len(episodes_on_page_after_click)}. "
                                 f"Assuming end of dynamic loading or a different pagination method.")
                    break # Break if no new elements appeared or count didn't increase
                
                page_iteration += 1 # Increment for the next iteration
            else:
                logging.info("  No 'Load More' button found or it's disabled. Ending pagination for this series.")
                break

    except Exception as e:
        logging.error(f"❌ Error getting series info for {series_url}: {e}")
        # Optional: Save a screenshot on error for debugging (won't work without X server for headless=True)
        try:
            screenshot_path = f"error_series_{sanitize_filename(series_name)}.png"
            page_browser.screenshot(path=screenshot_path)
            logging.error(f"  Screenshot saved to {screenshot_path}")
        except Exception as scr_e:
            logging.error(f"  Failed to save screenshot: {scr_e}")

    # Deduplicate episode_data based on episode_url at the very end
    # Note: We now primarily store direct audio URLs in 'episode_url' field
    unique_episode_data = []
    seen_keys = set()
    for item in episode_data:
        # Key for deduplication is the collected URL itself, which will now primarily be the direct audio URL
        key = item.get('episode_url') 
        if key and key not in seen_keys:
            unique_episode_data.append(item)
            seen_keys.add(key)
        elif key:
            logging.debug(f"  DEBUG: Skipping duplicate episode in final deduplication: {key}")

    logging.info(f"  Total unique episodes collected for '{series_name}': {len(unique_episode_data)}")
    return series_name, unique_episode_data

def get_audio_info(episode_urls_data: dict, podcast_series_name: str) -> dict or None:
    """
    Extracts detailed audio information (title, audio URL) directly from the provided URL,
    assuming it's a direct audio file. If not, it skips the item.
    """
    collected_url = episode_urls_data.get("episode_url") # This is the URL collected from the series page
    
    if not collected_url:
        logging.warning("    ⚠️ No URL found for an item. Skipping.")
        return None

    # STRICTLY check if the collected URL is a direct audio file (.mp3 or download domain)
    if ".mp3" in collected_url or "download-media.srf.ch" in collected_url:
        direct_audio_url = collected_url
        title = "Untitled Episode"
        
        # Try to infer title from the audio URL basename
        title_from_audio_basename = os.path.basename(direct_audio_url.split('?')[0]).replace('-', ' ').title()
        if title_from_audio_basename and not title_from_audio_basename.startswith("Aud"): # Avoid generic "Audio" titles
            title = title_from_audio_basename
        
        filename = sanitize_filename(os.path.basename(direct_audio_url.split("?")[0]))
        if not filename.endswith(".mp3"):
            filename += ".mp3"
        
        logging.info(f"    🎧 Using direct audio URL: '{title}'")
        return {
            "title": title,
            "podcast_name": podcast_series_name,
            "url": collected_url, # The original URL (which is the audio URL in this case)
            "audio_url": direct_audio_url,
            "filename": filename
        }
    else:
        # If it's not a direct MP3 link (e.g., it's an episode page link), we skip it as per your request
        logging.warning(f"    ⚠️ Skipping non-direct MP3 URL: {collected_url}")
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
                # Key for deduplication is the collected URL itself, which will now primarily be the direct audio URL
                key = item.get('episode_url') 
                if key and key not in seen_keys:
                    unique_episode_data.append(item)
                    seen_keys.add(key)
                elif key:
                    logging.debug(f"  DEBUG: Skipping duplicate episode in final deduplication: {key}")

            if not unique_episode_data:
                logging.warning(f"  No unique episodes found for series: '{podcast_name}' ({series_url}) after Playwright processing.")
                continue
                
            podcast_output_dir = os.path.join(OUTPUT_BASE_DIR, podcast_name)
            os.makedirs(podcast_output_dir, exist_ok=True)

            for ep_data in tqdm(unique_episode_data, desc=f"  Episodes in '{podcast_name}'", leave=False):
                info = get_audio_info(ep_data, podcast_name)
                if not info:
                    continue # Skip if audio info couldn't be retrieved (because it wasn't a direct MP3)
                
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
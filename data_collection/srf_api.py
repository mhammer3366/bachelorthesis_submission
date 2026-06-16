import requests
import xml.etree.ElementTree as ET
import os
from urllib.parse import urlparse
from pathlib import Path
import re

class SRFRSSDownloader:
    """
    Alternative Downloader für SRF Podcasts über RSS-Feeds
    (Für den Fall, dass die API kompliziert ist)
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Bekannte SRF Podcast RSS-URLs
        self.known_feeds = {
            'input': 'https://www.srf.ch/audio/input?format=podcast',
            'digital-podcast': 'https://www.srf.ch/audio/digital-podcast?format=podcast',
            'sounds': 'https://www.srf.ch/audio/sounds?format=podcast',
            'echo-der-zeit': 'https://www.srf.ch/audio/echo-der-zeit?format=podcast',
            'schnabelweid': 'https://www.srf.ch/audio/schnabelweid?format=podcast',
            'focus': 'https://www.srf.ch/audio/focus?format=podcast',
            'rendez-vous': 'https://www.srf.ch/audio/rendez-vous?format=podcast'
        }
    
    def get_rss_feed(self, rss_url):
        """RSS-Feed laden und parsen"""
        try:
            response = self.session.get(rss_url)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            
            # Channel-Info extrahieren
            channel = root.find('channel')
            if channel is None:
                print("Kein gültiger RSS-Feed gefunden")
                return None, []
            
            title = channel.find('title').text if channel.find('title') is not None else "Unbekannt"
            description = channel.find('description').text if channel.find('description') is not None else ""
            
            # Items (Episoden) extrahieren
            items = channel.findall('item')
            episodes = []
            
            for item in items:
                episode = {
                    'title': item.find('title').text if item.find('title') is not None else 'Unbekannt',
                    'description': item.find('description').text if item.find('description') is not None else '',
                    'pub_date': item.find('pubDate').text if item.find('pubDate') is not None else '',
                    'audio_url': None
                }
                
                # Audio-URL finden (enclosure oder link)
                enclosure = item.find('enclosure')
                if enclosure is not None:
                    episode['audio_url'] = enclosure.get('url')
                else:
                    # Alternative: Link-Element verwenden
                    link = item.find('link')
                    if link is not None:
                        episode['audio_url'] = link.text
                
                if episode['audio_url']:
                    episodes.append(episode)
            
            return {'title': title, 'description': description}, episodes
            
        except Exception as e:
            print(f"Fehler beim Laden des RSS-Feeds: {e}")
            return None, []
    
    def clean_filename(self, filename):
        """Dateiname für Dateisystem säubern"""
        # Ungültige Zeichen entfernen
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Mehrfache Leerzeichen durch eins ersetzen
        filename = re.sub(r'\s+', ' ', filename)
        # Länge begrenzen
        return filename.strip()[:200]
    
    def download_file(self, url, filepath):
        """Datei herunterladen mit Fortschrittsanzeige"""
        try:
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rDownload: {percent:.1f}%", end="", flush=True)
            
            print(f"\n✓ Gespeichert: {filepath}")
            return True
            
        except Exception as e:
            print(f"\n✗ Download-Fehler: {e}")
            return False
    
    def download_podcast(self, podcast_key, max_episodes=5, download_dir="srf_podcasts"):
        """Podcast über RSS-Feed herunterladen"""
        
        if podcast_key not in self.known_feeds:
            print(f"Unbekannter Podcast: {podcast_key}")
            print(f"Verfügbare Podcasts: {list(self.known_feeds.keys())}")
            return
        
        rss_url = self.known_feeds[podcast_key]
        print(f"Lade RSS-Feed: {rss_url}")
        
        # RSS-Feed laden
        podcast_info, episodes = self.get_rss_feed(rss_url)
        
        if not episodes:
            print("Keine Episoden gefunden")
            return
        
        print(f"\nPodcast: {podcast_info['title']}")
        print(f"Gefundene Episoden: {len(episodes)}")
        
        # Download-Verzeichnis erstellen
        podcast_dir = os.path.join(download_dir, self.clean_filename(podcast_info['title']))
        Path(podcast_dir).mkdir(parents=True, exist_ok=True)
        
        # Episoden herunterladen (neueste zuerst)
        episodes_to_download = episodes[:max_episodes]
        
        for i, episode in enumerate(episodes_to_download, 1):
            print(f"\n--- Episode {i}/{len(episodes_to_download)} ---")
            print(f"Titel: {episode['title']}")
            print(f"Datum: {episode['pub_date']}")
            
            if not episode['audio_url']:
                print("✗ Keine Audio-URL gefunden")
                continue
            
            # Dateiname generieren
            clean_title = self.clean_filename(episode['title'])
            
            # Dateiendung aus URL extrahieren
            parsed_url = urlparse(episode['audio_url'])
            extension = os.path.splitext(parsed_url.path)[1] or '.mp3'
            
            filename = f"{clean_title}{extension}"
            filepath = os.path.join(podcast_dir, filename)
            
            # Skip wenn bereits vorhanden
            if os.path.exists(filepath):
                print(f"✓ Bereits vorhanden: {filename}")
                continue
            
            # Download
            success = self.download_file(episode['audio_url'], filepath)
            
            if not success:
                # Aufräumen bei Fehler
                if os.path.exists(filepath):
                    os.remove(filepath)
    
    def list_available_podcasts(self):
        """Verfügbare Podcasts auflisten"""
        print("Verfügbare SRF Podcasts:")
        print("-" * 40)
        
        for key, url in self.known_feeds.items():
            print(f"• {key}")
            
            # Versuche Titel aus RSS zu laden
            try:
                info, _ = self.get_rss_feed(url)
                if info:
                    print(f"  '{info['title']}'")
            except:
                pass
            print()

def main():
    downloader = SRFRSSDownloader()
    
    print("=== SRF RSS Podcast Downloader ===\n")
    
    # Verfügbare Podcasts anzeigen
    downloader.list_available_podcasts()
    
    # Benutzer-Eingabe
    podcast_key = input("Welchen Podcast herunterladen? (Schlüsselwort eingeben): ").strip().lower()
    
    if podcast_key not in downloader.known_feeds:
        print(f"Unbekannter Podcast. Verfügbare: {list(downloader.known_feeds.keys())}")
        return
    
    try:
        max_episodes = input("Anzahl Episoden (Standard: 5): ").strip()
        max_episodes = int(max_episodes) if max_episodes.isdigit() else 5
        
        downloader.download_podcast(podcast_key, max_episodes)
        
    except KeyboardInterrupt:
        print("\nDownload abgebrochen")
    except Exception as e:
        print(f"Fehler: {e}")

if __name__ == "__main__":
    main()
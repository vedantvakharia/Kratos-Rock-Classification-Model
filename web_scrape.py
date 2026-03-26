"""
Shutterstock Image Downloader - Element Inspection Version
Downloads ALL images by inspecting element structure
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import requests
import os
import time
import re

class ShutterstockImageDownloader:
    def __init__(self, output_dir="downloaded_images"):
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✓ Created directory: {output_dir}\n")
        
        chrome_options = Options()
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        print("Opening browser...")
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        print("✓ Browser opened\n")
    
    def download_image(self, url, filename):
        """Download a single image"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://www.shutterstock.com/'
            }
            response = requests.get(url, headers=headers, timeout=15, stream=True)
            response.raise_for_status()
            
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = os.path.getsize(filepath)
            return True, file_size
        except Exception as e:
            return False, str(e)
    
    def inspect_and_download(self, search_url):
        """Inspect element structure and download all result images"""
        print(f"{'='*70}")
        print(f"Shutterstock Image Downloader - Inspecting Elements")
        print(f"{'='*70}\n")
        
        try:
            # Load page
            print(f"Loading: {search_url}")
            self.driver.get(search_url)
            
            print("\n⏳ Waiting for page to load...")
            time.sleep(5)
            
            # Scroll to load ALL images
            print("📜 Scrolling to load all images (this may take a while)...")
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            scroll_count = 0
            
            while True:
                # Scroll down
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)
                
                # Calculate new scroll height
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                scroll_count += 1
                print(f"  Scroll #{scroll_count} - Height: {new_height}px")
                
                # Break if no more content loads
                if new_height == last_height:
                    print("  ✓ Reached end of page")
                    break
                    
                last_height = new_height
                
                # Safety limit
                if scroll_count > 50:
                    print("  ⚠ Scroll limit reached (50 scrolls)")
                    break
            
            # Scroll back to top
            self.driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(2)
            
            # Now inspect elements to find image links
            print("\n🔍 Inspecting page structure for image links...\n")
            
            # Strategy: Find image elements from Shutterstock search results
            # Shutterstock typically uses <img> tags with specific attributes
            image_links = []
            
            # Find all <img> tags on the page
            img_elements = self.driver.find_elements(By.TAG_NAME, "img")
            print(f"Found {len(img_elements)} total images on page")
            
            for img in img_elements:
                try:
                    img_src = img.get_attribute('src') or img.get_attribute('data-src') or ''
                    alt_text = img.get_attribute('alt') or ''
                    
                    # Look for Shutterstock image URLs
                    # Shutterstock images are typically hosted on their CDN
                    if img_src and any(domain in img_src.lower() for domain in ['shutterstock.com', 'sstatic.net']):
                        # Exclude UI elements, logos, icons
                        if not any(x in img_src.lower() for x in ['logo', 'icon', 'avatar', 'badge', 'user', 'profile', 'sprite']):
                            # Try to get higher quality version
                            original_url = img_src
                            
                            # Shutterstock URL patterns - try to get larger versions
                            # Replace thumbnail size parameters with larger ones
                            original_url = re.sub(r'_(\d+)x(\d+)\.', '_1000x1000.', original_url)
                            original_url = re.sub(r'width=\d+', 'width=1000', original_url)
                            original_url = re.sub(r'height=\d+', 'height=1000', original_url)
                            
                            image_links.append({
                                'url': original_url,
                                'original_src': img_src,
                                'alt': alt_text
                            })
                except:
                    continue
            
            # Remove duplicates based on URL
            seen_urls = set()
            unique_images = []
            for img in image_links:
                if img['url'] not in seen_urls:
                    seen_urls.add(img['url'])
                    unique_images.append(img)
            
            print(f"\n✓ Found {len(unique_images)} unique images to download")
            
            if not unique_images:
                print("\n⚠ No images found. The page structure may be different.")
                print("Opening browser console for manual inspection...")
                input("Press Enter to close browser...")
                return
            
            # Show sample of what we found
            print(f"\nSample of images found (with alt text):")
            print("-" * 70)
            for i, img in enumerate(unique_images[:5], 1):
                print(f"{i}. Alt: {img['alt'][:50]}")
                print(f"   URL: {img['url'][:60]}...")
            if len(unique_images) > 5:
                print(f"... and {len(unique_images) - 5} more")
            print("-" * 70)
            
            # Ask for confirmation
            response = input(f"\nDownload ALL {len(unique_images)} images? (yes/no): ").strip().lower()
            
            if response != 'yes':
                print("\n❌ Download cancelled by user")
                return
            
            # Download all images
            print(f"\n📥 Downloading {len(unique_images)} images...\n")
            
            successful = 0
            failed = 0
            
            for idx, img_data in enumerate(unique_images, 1):
                url = img_data['url']
                
                # Generate filename
                base_name = os.path.basename(url.split('?')[0])
                if not base_name or '.' not in base_name:
                    # Extract from URL or use index
                    match = re.search(r'/([^/]+\.(jpg|jpeg|png|webp))', url)
                    base_name = match.group(1) if match else f"image_{idx}.jpg"
                
                filename = f"{idx:04d}_{base_name}"
                filename = re.sub(r'[^\w\-_\.]', '_', filename)
                
                print(f"[{idx}/{len(unique_images)}] {filename}")
                success, result = self.download_image(url, filename)
                
                if success:
                    print(f"  ✓ {result / 1024:.1f} KB")
                    successful += 1
                else:
                    print(f"  ✗ {str(result)[:50]}")
                    failed += 1
                
                # Small delay to be polite
                time.sleep(0.3)
            
            print(f"\n{'='*70}")
            print(f"Download Complete!")
            print(f"{'='*70}")
            print(f"✓ Successfully downloaded: {successful}")
            print(f"✗ Failed: {failed}")
            print(f"📁 Location: {os.path.abspath(self.output_dir)}")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            print("\n👋 Closing browser in 5 seconds...")
            time.sleep(5)
            self.driver.quit()
            print("✓ Done!\n")


if __name__ == "__main__":
    SEARCH_URL = "https://www.shutterstock.com/search/chert?image_type=photo"
    OUTPUT_DIR = "chert_rock_images"
    
    downloader = ShutterstockImageDownloader(output_dir=OUTPUT_DIR)
    downloader.inspect_and_download(search_url=SEARCH_URL)
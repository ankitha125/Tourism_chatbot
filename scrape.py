import requests
from bs4 import BeautifulSoup
import pandas as pd
import hashlib
import os
import time
import re
from urllib.parse import urljoin, urlparse
import pdfplumber
import pytesseract
from pdf2image import convert_from_path

# --- CONFIGURATION ---
BASE_URL = "https://arunachaltourism.com/"
MAX_PAGES = None
OUTPUT_CSV = "arunachal_tourism_final_cleaned.csv"
PDF_DIR = "downloaded_pdfs"

# Tesseract Setup (Update path if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = None

if not os.path.exists(PDF_DIR):
    os.makedirs(PDF_DIR)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def generate_unique_id(url):
    return hashlib.md5(url.encode()).hexdigest()[:10]


def download_pdf(pdf_url):
    try:
        parsed_url = urlparse(pdf_url)
        filename = os.path.basename(parsed_url.path)
        if not filename.lower().endswith(".pdf"):
            filename = f"{generate_unique_id(pdf_url)}.pdf"
        local_path = os.path.join(PDF_DIR, filename)

        response = requests.get(pdf_url, headers=HEADERS, stream=True, timeout=20)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return local_path
    except Exception:
        return None


def process_pdf(pdf_path):
    text_content = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_content.append(text)
        final_text = "\n".join(text_content)

        # OCR Fallback
        if len(final_text) < 50:
            images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
            ocr_text = [pytesseract.image_to_string(img) for img in images]
            final_text = "\n".join(ocr_text)
            method = "OCR"
        else:
            method = "Standard"
        return final_text, method
    except Exception as e:
        # IMPROVEMENT: Return the actual error message
        return f"Error reading PDF: {str(e)}", "Failed"


# --- MAIN LOGIC FOR IMAGES & TEXT ---
def scrape_web_page(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # 1. CLEANER TEXT EXTRACTION (Fixes Duplicates)
        # Remove script and style elements first
        for script in soup(["script", "style", "nav", "footer"]):
            script.extract()

        # Get text with separators (much cleaner than find_all)
        text = soup.get_text(separator="\n")
        # Clean up empty lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        content_text = "\n".join(lines)

        # 2. IMAGE EXTRACTION (Standard + Background Regex)
        image_urls = set()

        # Type A: Standard <img> tags
        for img in soup.find_all("img", src=True):
            full_img_url = urljoin(url, img["src"])
            image_urls.add(full_img_url)

        # Type B: Background Images in <div> (Regex)
        divs_with_style = soup.find_all("div", style=True)
        for div in divs_with_style:
            style_content = div["style"]
            match = re.search(r'url\([\'"]?(.*?)[\'"]?\)', style_content)
            if match:
                bg_url = match.group(1)
                full_bg_url = urljoin(url, bg_url)
                image_urls.add(full_bg_url)

        final_image_string = "\n".join(list(image_urls))

        # 3. META TAGS (Expanded)
        meta_title = soup.title.get_text(strip=True) if soup.title else ""

        desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find(
            "meta", property="og:description"
        )
        meta_desc = desc_tag["content"] if desc_tag else ""

        # IMPROVEMENT: Added Keywords
        kw_tag = soup.find("meta", attrs={"name": "keywords"})
        meta_kw = kw_tag["content"] if kw_tag else ""

        og_img_tag = soup.find("meta", property="og:image")
        meta_og_image = og_img_tag["content"] if og_img_tag else ""

        return {
            "content": content_text,
            "image_urls": final_image_string,
            "meta_title": meta_title,
            "meta_desc": meta_desc,
            "meta_kw": meta_kw,
            "meta_og_image": meta_og_image,
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


# --- RUNNER ---
print("--- Starting Enhanced Scraper ---")
try:
    homepage_soup = BeautifulSoup(
        requests.get(BASE_URL, headers=HEADERS).text, "html.parser"
    )
    all_links = homepage_soup.find_all("a", href=True)
except Exception:
    all_links = []

data_list = []
processed_urls = set()

for index, link in enumerate(all_links):
    if MAX_PAGES and len(data_list) >= MAX_PAGES:
        break

    href = link["href"]
    full_url = urljoin(BASE_URL, href)

    # IMPROVEMENT: Better Filters (skips mailto/tel)
    if "arunachaltourism.com" not in full_url or full_url in processed_urls:
        continue
    if any(
        x in full_url.lower() for x in [".jpg", ".png", "javascript", "mailto:", "tel:"]
    ):
        continue

    processed_urls.add(full_url)
    print(f"[{len(data_list) + 1}] Processing: {full_url}")

    if full_url.lower().endswith(".pdf"):
        local_path = download_pdf(full_url)
        if local_path:
            text, method = process_pdf(local_path)
            data_list.append(
                {
                    "Parent_URL": BASE_URL,
                    "Child_URL": full_url,
                    "Content_Type": "PDF",
                    "Meta_Title": "PDF Document",
                    "Meta_Description": f"Method: {method}",
                    "Meta_Keywords": "N/A",
                    "OG_Image": "N/A",
                    "Image_URLs": "N/A",
                    "Extracted_Content": text,
                }
            )
    else:
        content = scrape_web_page(full_url)
        if content:
            data_list.append(
                {
                    "Parent_URL": BASE_URL,
                    "Child_URL": full_url,
                    "Content_Type": "Web Page",
                    "Meta_Title": content["meta_title"],
                    "Meta_Description": content["meta_desc"],
                    "Meta_Keywords": content["meta_kw"],
                    "OG_Image": content["meta_og_image"],
                    "Image_URLs": content["image_urls"],
                    "Extracted_Content": content["content"],
                }
            )
    time.sleep(1)

# --- SAVE ---
if data_list:
    df = pd.DataFrame(data_list)
    cols = [
        "Parent_URL",
        "Child_URL",
        "Content_Type",
        "Meta_Title",
        "Meta_Description",
        "Meta_Keywords",
        "OG_Image",
        "Image_URLs",
        "Extracted_Content",
    ]
    # Ensure all columns exist before selecting
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols]

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved! Check {OUTPUT_CSV}")

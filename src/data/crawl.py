from typing import Tuple, List, Optional
import validators
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging
import requests
import json
import os


def validate_wiki_url(url: str) -> Tuple[bool, str]:
    """
    Validate if the given URL is a valid Wikipedia URL
    
    Args:
        url (str): URL to validate
        
    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    if not validators.url(url):
        return False, "Invalid URL format"
    
    wiki_domains = ["wikipedia.org", "en.wikipedia.org", "vi.wikipedia.org", "de.wikipedia.org"]
    if not any(domain in url.lower() for domain in wiki_domains):
        return False, "Not a Wikipedia URL"
    return True, url


def fetch_content(url):
    try:
        # Gửi request đến URL
        response = requests.get(url)
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        
        # Parse HTML với BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Lấy tiêu đề
        title = soup.find('h1', {'id': 'firstHeading'}).text
        
        # Lấy nội dung chính
        content_div = soup.find('div', {'id': 'mw-content-text'})
        paragraphs = content_div.find_all('p')
        
        # Lấy text từ các đoạn văn
        content = []
        for p in paragraphs:
            text = p.get_text()
            if text: 
                content.append(text)
        
        
        return {
            'title': title,
            'content': content,
            'status': 'success'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }
        

stop_words = set([
    "và", "là", "của", "có", "cho", "một", "các", "những", "được", "trong", "với", "đã", "từ", "sẽ", "rằng", "vì", "ở", "đến", "khi", "này", "ra", "như", "nên", "thì", "bị", "bởi", "đó", "nào", "sau", "trên", "vẫn", "vào", "hơn", "nữa", "đây", "để"
])

def clean_and_process_text(text: str) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    text = re.sub(r'\[\d+\]', '', text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-zA-ZÀ-ỹà-ỹ0-9\s]', ' ', text)
    text = text.lower()
    tokens = text.split()
    processed_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(processed_tokens)
    
def process_wikipedia_url(url: str) -> dict:
    """
    Main pipeline to process Wikipedia URL
    
    Args:
        url (str): Wikipedia URL
        
    Returns:
        dict: Processing results and status
    """
    # Step 1 & 2: Validate URL
    is_valid, message = validate_wiki_url(url)
    if not is_valid:
        return {"status": "error", "message": message}
   
    # Step 3: Fetch content
    raw_content = fetch_content(url)
    if not raw_content:
        return {"status": "error", "message": "Failed to fetch content"}
    
    
    
    # Step 5: Process text (trả về list)
   
    processed_list = [clean_and_process_text(paragraph) for paragraph in raw_content['content']]
    if not any(processed_list):
        return {"status": "error", "message": "Failed to process text"}
    
    # Step 6 & 7: Return processed text for next pipeline
    return {
        "status": "success",
        "raw_content": raw_content,
        "processed_text": processed_list
    }

def crawl_urls_from_file(file_path: str, save_folder: str):
    os.makedirs(save_folder, exist_ok=True)
    with open(file_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("//")]

    for url in urls:
        print(f"Đang xử lý: {url}")
        try:
            result = process_wikipedia_url(url)
            if result['status'] == 'success':
                # Tạo tên file từ title
                filename = re.sub(r'[^\w\s]', '', result['raw_content']['title'])
                filename = re.sub(r'\s+', '_', filename) + ".json"
                save_path = os.path.join(save_folder, filename)
                with open(save_path, 'w', encoding='utf-8') as f_json:
                    json.dump(result, f_json, ensure_ascii=False, indent=2)
                print(f"Đã lưu: {save_path}")
            else:
                print(f"Lỗi: {result['message']}")
        except Exception as e:
            print(f"Lỗi không mong muốn với URL {url}: {e}")  

if __name__ == "__main__":

    crawl_urls_from_file("wiki_source/wiki_urls.txt", "data_crawl")
   
    

    
    
    # # Thực hiện scraping
    # print("Đang scrape dữ liệu...")
    # result = fetch_content(url)
    
    # # Hiển thị kết quả
    # if result['status'] == 'success':
    #     print("\n" + "="*50)
    #     print(f"TIÊU ĐỀ: {result['title']}")
    #     print("="*50 + "\n")
        
    #     print("NỘI DUNG:")
    #     for i, paragraph in enumerate(result['content'], 1):
    #         print(f"{i}. {paragraph}\n")
            
    #     # Lưu kết quả vào file JSON (tùy chọn)
    #     save_option = input("Bạn có muốn lưu kết quả vào file JSON không? (y/n): ")
    #     if save_option.lower() == 'y':
    #         filename = f"{result['title'].replace(' ', '_')}.json"
    #         with open(filename, 'w', encoding='utf-8') as f:
    #             json.dump(result, f, ensure_ascii=False, indent=2)
    #         print(f"Đã lưu kết quả vào file {filename}")
    # else:
    #     print(f"Lỗi: {result['message']}")

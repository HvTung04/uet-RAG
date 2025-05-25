"""
Script setup môi trường cho hệ thống đánh giá RAG
"""

import subprocess
import sys
import os

def install_requirements():
    """Cài đặt các dependencies"""
    print("🔧 Cài đặt dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "eval_requirements.txt"])
        print("✅ Dependencies đã được cài đặt thành công!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi cài đặt dependencies: {e}")
        return False
    return True

def download_nltk_data():
    """Download NLTK data cần thiết"""
    print("📥 Download NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("✅ NLTK data đã được download thành công!")
    except Exception as e:
        print(f"❌ Lỗi download NLTK data: {e}")
        return False
    return True

def download_sentence_transformer():
    """Download sentence transformer model"""
    print("🤖 Download sentence transformer model...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence transformer model đã được download thành công!")
    except Exception as e:
        print(f"❌ Lỗi download sentence transformer: {e}")
        return False
    return True

def check_config():
    """Kiểm tra file config"""
    print("📋 Kiểm tra cấu hình...")
    
    if not os.path.exists("config.yaml"):
        print("⚠️  Không tìm thấy config.yaml")
        return False
    
    if not os.path.exists("data/wiki_data"):
        print("⚠️  Không tìm thấy thư mục data/wiki_data")
        return False
    
    print("✅ Cấu hình OK!")
    return True

def main():
    """Chạy setup"""
    print("🚀 SETUP HỆ THỐNG ĐÁNH GIÁ RAG")
    print("=" * 50)
    
    # Kiểm tra Python version
    if sys.version_info < (3, 7):
        print("❌ Cần Python 3.7 trở lên")
        return
    
    print(f"✅ Python version: {sys.version}")
    
    # Cài đặt dependencies
    if not install_requirements():
        return
    
    # Download NLTK data
    if not download_nltk_data():
        return
    
    # Download sentence transformer
    if not download_sentence_transformer():
        return
    
    # Kiểm tra config
    config_ok = check_config()
    
    print("\n" + "=" * 50)
    if config_ok:
        print("🎉 Setup hoàn thành! Hệ thống đánh giá đã sẵn sàng.")
        print("\nCách sử dụng:")
        print("  python app.py --evaluate")
        print("  python eval.py --wiki_data data/wiki_data")
        print("  python demo_eval.py")
    else:
        print("⚠️  Setup hoàn thành nhưng cần kiểm tra cấu hình.")
        print("Đảm bảo có file config.yaml và thư mục data/wiki_data")

if __name__ == "__main__":
    main() 
"""
Upload Sticker to WhatsApp and Get Media ID
Run this script once to upload your sticker and get the permanent Media ID
"""

import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")

def upload_sticker_to_whatsapp(sticker_path):
    """
    Upload a sticker to WhatsApp and get its Media ID
    
    Args:
        sticker_path: Path to your sticker file (must be .webp format, 512x512px)
    
    Returns:
        Media ID string if successful, None if failed
    """
    
    print("\n" + "="*60)
    print("🚀 UPLOADING STICKER TO WHATSAPP")
    print("="*60)
    
    # Check if file exists
    if not os.path.exists(sticker_path):
        print(f"❌ Error: File not found: {sticker_path}")
        return None
    
    # Check file extension
    if not sticker_path.lower().endswith('.webp'):
        print("⚠️ Warning: WhatsApp stickers should be in WebP format")
        print("   Continuing anyway, but it may fail...")
    
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/media"
    
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}"
    }
    
    # Open the file and prepare for upload
    with open(sticker_path, 'rb') as f:
        files = {
            'file': (os.path.basename(sticker_path), f, 'image/webp')
        }
        
        data = {
            'messaging_product': 'whatsapp',
            'type': 'image/webp'
        }
        
        try:
            print(f"\n📤 Uploading: {sticker_path}")
            print(f"📏 File size: {os.path.getsize(sticker_path) / 1024:.2f} KB")
            
            response = requests.post(url, headers=headers, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                media_id = result.get('id')
                
                print("\n✅ SUCCESS! Sticker uploaded!")
                print("="*60)
                print(f"\n🎯 YOUR MEDIA ID: {media_id}")
                print("\n📝 Add this to your .env file:")
                print(f"\nCRAYHEALTH_STICKER_ID={media_id}")
                print("\n" + "="*60)
                
                return media_id
            else:
                error_data = response.json()
                print(f"\n❌ Upload failed!")
                print(f"Status Code: {response.status_code}")
                print(f"Error: {error_data}")
                
                # Common error explanations
                if response.status_code == 400:
                    print("\n💡 Possible reasons:")
                    print("   - File is not in WebP format")
                    print("   - File is not 512x512 pixels")
                    print("   - File size is over 100KB")
                elif response.status_code == 401:
                    print("\n💡 Authentication error - check your WHATSAPP_TOKEN")
                elif response.status_code == 403:
                    print("\n💡 Permission error - check your WhatsApp Business account permissions")
                
                return None
                
        except Exception as e:
            print(f"\n❌ Exception occurred: {e}")
            return None


def convert_image_to_sticker(input_path, output_path="crayhealth-sticker.webp"):
    """
    Convert any image to WhatsApp sticker format (WebP, 512x512px)
    Requires: pip install Pillow
    """
    try:
        from PIL import Image
        
        print(f"\n🔄 Converting {input_path} to sticker format...")
        
        # Open image
        img = Image.open(input_path)
        
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Resize to 512x512
        img = img.resize((512, 512), Image.LANCZOS)
        
        # Save as WebP
        img.save(output_path, 'WEBP', quality=95)
        
        file_size = os.path.getsize(output_path) / 1024
        print(f"✅ Converted successfully!")
        print(f"   Output: {output_path}")
        print(f"   Size: {file_size:.2f} KB")
        
        if file_size > 100:
            print(f"⚠️ Warning: File size ({file_size:.2f} KB) exceeds WhatsApp's 100KB limit")
            print("   Try reducing quality or image complexity")
        
        return output_path
        
    except ImportError:
        print("❌ Error: Pillow library not installed")
        print("   Install it with: pip install Pillow")
        return None
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return None


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════╗
║        CRAYHEALTH STICKER UPLOADER                         ║
║        Upload your sticker and get Media ID                ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Check if credentials are configured
    if not PHONE_NUMBER_ID or not WHATSAPP_TOKEN:
        print("❌ Error: Missing WhatsApp credentials in .env file")
        print("\nMake sure you have:")
        print("   PHONE_NUMBER_ID=your_phone_number_id")
        print("   WHATSAPP_TOKEN=your_whatsapp_token")
        exit(1)
    
    print(f"✅ Phone Number ID: {PHONE_NUMBER_ID}")
    print(f"✅ Token configured: {'*' * 20}")
    
    # Option 1: Upload existing WebP sticker
    print("\n" + "="*60)
    print("OPTION 1: Upload existing WebP sticker")
    print("="*60)
    sticker_path = input("\nEnter path to your .webp sticker file (or press Enter to skip): ").strip()
    
    if sticker_path and os.path.exists(sticker_path):
        media_id = upload_sticker_to_whatsapp(sticker_path)
        if media_id:
            print(f"\n🎉 Done! Use this Media ID: {media_id}")
            exit(0)
    
    # Option 2: Convert image to sticker format first
    print("\n" + "="*60)
    print("OPTION 2: Convert image to sticker format, then upload")
    print("="*60)
    image_path = input("\nEnter path to your image file (PNG/JPG/JPEG): ").strip()
    
    if image_path and os.path.exists(image_path):
        # Convert to sticker format
        sticker_path = convert_image_to_sticker("C:/Users/msi/Downloads/crayhealth-logo.jpeg")
        
        if sticker_path:
            # Upload the converted sticker
            media_id = upload_sticker_to_whatsapp(sticker_path)
            
            if media_id:
                print(f"\n🎉 Done! Use this Media ID: {media_id}")
                print(f"\n💾 Sticker saved as: {sticker_path}")
            else:
                print(f"\n💾 Sticker saved as: {sticker_path}")
                print("   You can try uploading it manually through Business Manager")
    else:
        print("\n❌ No valid file path provided")
        print("\nPlease run the script again with a valid image file path")
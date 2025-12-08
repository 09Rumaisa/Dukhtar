"""
Script to upload doctor profile images from local static folder to AWS S3
Run this once to migrate images to S3
"""

import os
from aws_s3_utils import get_s3_manager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def upload_local_images_to_s3():
    """Upload all doctor images from static/doctors_profile to S3"""
    
    # Initialize S3 manager
    s3 = get_s3_manager()
    
    # Local directory containing doctor images
    local_dir = 'static/doctors_profile'
    
    if not os.path.exists(local_dir):
        print(f"❌ Directory {local_dir} not found")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(local_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"❌ No image files found in {local_dir}")
        return
    
    print(f"Found {len(image_files)} images to upload")
    print("-" * 50)
    
    uploaded_count = 0
    failed_count = 0
    
    for filename in image_files:
        try:
            local_path = os.path.join(local_dir, filename)
            
            # Upload to S3
            print(f"Uploading {filename}...")
            url = s3.upload_from_local_file(
                local_path=local_path,
                destination_key=f"doctors/{filename}",
                bucket_type='images'
            )
            
            print(f"✓ Uploaded: {url}")
            uploaded_count += 1
            
        except Exception as e:
            print(f"❌ Failed to upload {filename}: {e}")
            failed_count += 1
    
    print("-" * 50)
    print(f"\n✓ Upload complete!")
    print(f"  - Successfully uploaded: {uploaded_count}")
    print(f"  - Failed: {failed_count}")
    print(f"\nImages are now available at:")
    print(f"https://dukhtar-doctorimages.s3.amazonaws.com/doctors/[filename]")

if __name__ == "__main__":
    print("=" * 50)
    print("Doctor Images S3 Upload Script")
    print("=" * 50)
    print()
    
    # Check if AWS credentials are configured
    if not os.getenv('AWS_ACCESS_KEY_ID'):
        print("❌ AWS_ACCESS_KEY_ID not found in environment variables")
        print("Please ensure your .env file is configured correctly")
        exit(1)
    
    if not os.getenv('AWS_SECRET_ACCESS_KEY'):
        print("❌ AWS_SECRET_ACCESS_KEY not found in environment variables")
        print("Please ensure your .env file is configured correctly")
        exit(1)
    
    print("✓ AWS credentials found")
    print()
    
    # Run upload
    upload_local_images_to_s3()

"""
AWS S3 utility functions for file storage
Handles uploads, downloads, and file management
"""

import boto3
from botocore.exceptions import ClientError
import os
import uuid
from datetime import datetime

class S3Manager:
    def __init__(self):
        """Initialize S3 client with credentials from environment"""
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        
        # Bucket names from environment
        self.images_bucket = os.getenv('AWS_S3_IMAGES_BUCKET', 'dukhtar-doctorimages')
        self.audio_bucket = os.getenv('AWS_S3_AUDIO_BUCKET', 'dukhtar-audio-temp')
    
    def upload_doctor_image(self, file_data, filename, content_type='image/jpeg'):
        """
        Upload doctor profile image to S3
        
        Args:
            file_data: Binary file data or file-like object
            filename: Name for the file
            content_type: MIME type
            
        Returns:
            Public URL of uploaded file
        """
        try:
            # Create unique path
            object_key = f"doctors/{filename}"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.images_bucket,
                Key=object_key,
                Body=file_data,
                ContentType=content_type,
                CacheControl='max-age=31536000',  # Cache for 1 year
            )
            
            # Generate public URL
            url = f"https://{self.images_bucket}.s3.amazonaws.com/{object_key}"
            
            print(f"✓ Uploaded doctor image: {url}")
            return url
            
        except ClientError as e:
            print(f"❌ Error uploading doctor image: {e}")
            raise
    
    def upload_audio_file(self, audio_data, filename=None):
        """
        Upload temporary audio file to S3
        Auto-deletes after 1 day (via lifecycle rule)
        
        Args:
            audio_data: Binary audio data
            filename: Optional filename (generates UUID if not provided)
            
        Returns:
            Public URL of uploaded file
        """
        try:
            # Generate filename if not provided
            if not filename:
                filename = f"speech_{uuid.uuid4().hex[:8]}.mp3"
            
            object_key = f"tts/{filename}"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.audio_bucket,
                Key=object_key,
                Body=audio_data,
                ContentType='audio/mpeg',
                CacheControl='max-age=86400',  # Cache for 1 day
            )
            
            # Generate public URL
            url = f"https://{self.audio_bucket}.s3.amazonaws.com/{object_key}"
            
            print(f"✓ Uploaded audio file: {url}")
            return url
            
        except ClientError as e:
            print(f"❌ Error uploading audio: {e}")
            raise
    
    def delete_audio_file(self, filename):
        """
        Delete audio file from S3
        
        Args:
            filename: Name of file to delete (or full URL)
        """
        try:
            # Handle both full URLs and just filenames
            if filename.startswith('http'):
                # Extract key from URL
                object_key = filename.split('.com/')[-1]
            else:
                object_key = f"tts/{filename}"
            
            self.s3_client.delete_object(
                Bucket=self.audio_bucket,
                Key=object_key
            )
            
            print(f"✓ Deleted audio file: {object_key}")
            
        except ClientError as e:
            print(f"❌ Error deleting audio: {e}")
            # Don't raise - file might already be deleted
    
    def upload_from_local_file(self, local_path, destination_key, bucket_type='images'):
        """
        Upload existing local file to S3
        
        Args:
            local_path: Path to local file
            destination_key: S3 object key (path in bucket)
            bucket_type: 'images' or 'audio'
            
        Returns:
            Public URL of uploaded file
        """
        try:
            bucket = self.images_bucket if bucket_type == 'images' else self.audio_bucket
            
            # Determine content type from file extension
            content_type = 'image/jpeg'
            if local_path.endswith('.png'):
                content_type = 'image/png'
            elif local_path.endswith('.mp3'):
                content_type = 'audio/mpeg'
            
            # Upload file
            with open(local_path, 'rb') as f:
                self.s3_client.put_object(
                    Bucket=bucket,
                    Key=destination_key,
                    Body=f,
                    ContentType=content_type
                )
            
            # Generate public URL
            url = f"https://{bucket}.s3.amazonaws.com/{destination_key}"
            
            print(f"✓ Uploaded local file: {url}")
            return url
            
        except ClientError as e:
            print(f"❌ Error uploading local file: {e}")
            raise
    
    def list_bucket_files(self, bucket_type='images', prefix=''):
        """
        List files in bucket
        
        Args:
            bucket_type: 'images' or 'audio'
            prefix: Filter by prefix (folder path)
            
        Returns:
            List of file keys
        """
        try:
            bucket = self.images_bucket if bucket_type == 'images' else self.audio_bucket
            
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
            
        except ClientError as e:
            print(f"❌ Error listing files: {e}")
            return []
    
    def get_bucket_size(self, bucket_type='images'):
        """
        Get total size of files in bucket (in MB)
        
        Args:
            bucket_type: 'images' or 'audio'
            
        Returns:
            Size in MB
        """
        try:
            bucket = self.images_bucket if bucket_type == 'images' else self.audio_bucket
            
            response = self.s3_client.list_objects_v2(Bucket=bucket)
            
            if 'Contents' in response:
                total_size = sum(obj['Size'] for obj in response['Contents'])
                return total_size / (1024 * 1024)  # Convert to MB
            return 0
            
        except ClientError as e:
            print(f"❌ Error getting bucket size: {e}")
            return 0


# Global instance
s3_manager = None

def get_s3_manager():
    """Get or create S3 manager instance"""
    global s3_manager
    if s3_manager is None:
        s3_manager = S3Manager()
    return s3_manager

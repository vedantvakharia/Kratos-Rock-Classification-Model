"""
Background Removal Preprocessing Script
Removes backgrounds from all rock images in Dataset folder
and saves processed images to Dataset1 folder.
"""

import os
from pathlib import Path
from PIL import Image
from rembg import remove
from tqdm import tqdm

# Paths
SOURCE_DIR = r"C:\Users\Vedant\Downloads\Rock Classification Models\Dataset"
OUTPUT_DIR = r"C:\Users\Vedant\Downloads\Rock Classification Models\Dataset1"

def get_all_images(root_dir):
    """Get all image files from the dataset directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    images = []
    
    for class_folder in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_folder)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if Path(img_file).suffix.lower() in image_extensions:
                    images.append((class_folder, img_file))
    
    return images

def process_images():
    """Process all images to remove backgrounds."""
    
    # Get all images
    print(f"Scanning images in {SOURCE_DIR}...")
    images = get_all_images(SOURCE_DIR)
    print(f"Found {len(images)} images to process.\n")
    
    if len(images) == 0:
        print("No images found. Please check the Dataset path.")
        return
    
    # Process each image
    success_count = 0
    error_count = 0
    
    for class_name, img_name in tqdm(images, desc="Removing backgrounds"):
        try:
            # Input and output paths
            input_path = os.path.join(SOURCE_DIR, class_name, img_name)
            output_class_dir = os.path.join(OUTPUT_DIR, class_name)
            os.makedirs(output_class_dir, exist_ok=True)
            
            # Change extension to PNG for transparency support
            output_name = Path(img_name).stem + ".png"
            output_path = os.path.join(output_class_dir, output_name)
            
            # Skip if already processed
            if os.path.exists(output_path):
                success_count += 1
                continue
            
            # Load image
            with Image.open(input_path) as img:
                # Convert to RGB if needed (for JPEGs)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Remove background
                output = remove(img)
                
                # Save as PNG with transparency
                output.save(output_path, 'PNG')
                success_count += 1
                
        except Exception as e:
            error_count += 1
            tqdm.write(f"Error processing {class_name}/{img_name}: {str(e)}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {success_count}/{len(images)}")
    print(f"❌ Errors: {error_count}/{len(images)}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    print("="*60)
    print("Background Removal Preprocessing")
    print("="*60)
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Error: Source directory not found: {SOURCE_DIR}")
        print("Please make sure the Dataset folder exists.")
    else:
        process_images()

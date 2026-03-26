import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import os
import sys

# --- 1. CONFIGURATION ---
# These are the 16 folders from your screenshot.
# We put them in a list and sort them exactly how PyTorch does (Case-Sensitive).
RAW_CLASS_NAMES = [
    "(Mudstone) Shale",
    "Basalt",
    "Basalt olivine",
    "Chert",
    "Clay",
    "Coal",
    "Conglomerate",
    "Diatomite",
    "Granite",
    "Gypsum",
    "Limestone",
    "Marble",
    "Quartz",
    "Quartzite",
    "Sandstone"
]

# CRITICAL: PyTorch sorts folders A-Z (Case Sensitive) during training.
# We must sort this list here to match the internal ID the model gave each rock.
CLASS_NAMES = sorted(RAW_CLASS_NAMES)

MODEL_PATH = "best_model.pth"

# --- 2. MODEL SETUP ---
def get_model(num_classes):
    print(f"Building model for {num_classes} classes...")
    model = torchvision.models.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model

def get_inference_transforms():
    return transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(380),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# --- 3. PREDICTION FUNCTION ---
def predict_rock(image_path, model, device):
    transform = get_inference_transforms()
    
    try:
        # Open and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        return CLASS_NAMES[predicted_idx.item()], confidence.item() * 100
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, 0

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    # Force the script to use the hardcoded classes
    print(f"Loaded {len(CLASS_NAMES)} classes.")
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Load Model ONCE
    model = get_model(len(CLASS_NAMES))
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)
            model.eval()
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            sys.exit()
    else:
        print(f"'{MODEL_PATH}' not found!")
        sys.exit()

    print("\n" + "="*40)
    print("  ROCK CLASSIFICATION INFERENCE")
    print("  Drag and drop an image file below.")
    print("="*40)

    while True:
        try:
            user_input = input("\nPath to image (or 'q' to quit): ").strip()
        except KeyboardInterrupt:
            break

        if user_input.lower() in ['q', 'quit', 'exit']:
            break
            
        # --- FIX FOR VS CODE / POWERSHELL ---
        # 1. Remove the leading '&' if present
        if user_input.startswith('&'):
            user_input = user_input[1:]
        
        # 2. Clean up whitespace and quotes
        user_input = user_input.strip().strip('"').strip("'")
        
        if not user_input:
            continue
            
        if not os.path.exists(user_input):
            print("File not found. Please try again.")
            continue

        # --- PREDICT ---
        transform = get_inference_transforms()
        try:
            image = Image.open(user_input).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get Top 3 Predictions
                top_prob, top_idx = torch.topk(probabilities, 3)
                
            # Print Results
            print(f"\nImage: {os.path.basename(user_input)}")
            print("-" * 30)
            
            # Show Top 1 (Winner)
            winner_idx = top_idx[0][0].item()
            winner_prob = top_prob[0][0].item() * 100
            print(f"WINNER: {CLASS_NAMES[winner_idx]} ({winner_prob:.2f}%)")
            
            # Show Runners Up
            print("\n   Runners Up:")
            for i in range(1, 3):
                idx = top_idx[0][i].item()
                prob = top_prob[0][i].item() * 100
                print(f"   {i+1}. {CLASS_NAMES[idx]}: {prob:.2f}%")
                
        except Exception as e:
            print(f"Error processing image: {e}")
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import math

class CreditCardNet(nn.Module):
    def __init__(self):
        super(CreditCardNet, self).__init__()
        # First conv block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Second conv block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Third conv block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Calculate the size of the flattened features
        self._to_linear = None
        self._get_conv_output_size()
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def _get_conv_output_size(self):
        # Create a dummy input to calculate the size
        x = torch.randn(1, 1, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        self._to_linear = x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CreditCardDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Add morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return thresh

def find_card_contour(image):
    # Find all contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour by area
    if not contours:
        return None
    
    card_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to get a rectangle
    peri = cv2.arcLength(card_contour, True)
    
    # Try different epsilon values to get a better approximation
    for eps in [0.02, 0.04, 0.06]:
        approx = cv2.approxPolyDP(card_contour, eps * peri, True)
        if len(approx) == 4:
            return approx
    
    # If we still don't have 4 points, try to get the minimum area rectangle
    rect = cv2.minAreaRect(card_contour)
    box = cv2.boxPoints(rect)
    return np.int32(box)

def order_points(pts):
    # Initialize a list of coordinates that will be ordered
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Sum of coordinates
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    
    # Difference of coordinates
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    
    return rect

def four_point_transform(image, pts):
    # Obtain a consistent order of the points
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Construct set of destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def load_and_prepare_dataset(dataset_path):
    images = []
    labels = []
    
    # Load images from each digit folder
    for digit in range(10):
        digit_path = os.path.join(dataset_path, str(digit))
        for img_name in os.listdir(digit_path):
            img_path = os.path.join(digit_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                # Resize to 32x32 and convert to grayscale
                img = cv2.resize(img, (32, 32))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(img)
                labels.append(digit)
    
    return np.array(images), np.array(labels)

def train_model():
    # Load and prepare dataset
    X, y = load_and_prepare_dataset('Credit Card Number Dataset')
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(X).unsqueeze(1) / 255.0
    y = torch.LongTensor(y)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Enhanced data augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = CreditCardDataset(X_train, y_train, transform=train_transform)
    test_dataset = CreditCardDataset(X_test, y_test, transform=test_transform)
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    
    # Create and train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CreditCardNet().to(device)
    
    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler - removed verbose parameter
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    criterion = nn.CrossEntropyLoss()
    
    # Training parameters
    num_epochs = 50
    best_accuracy = 0
    patience = 10
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        val_loss = val_loss / len(test_loader)
        
        # Update learning rate
        scheduler.step(val_accuracy)
        
        # Early stopping
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
            }, 'best_credit_card_model.pth')
        else:
            patience_counter += 1
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.2f}s)')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    # Load best model
    checkpoint = torch.load('best_credit_card_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def detect_card_angle(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return 0
    
    # Calculate angles of all lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            angle = 90
        else:
            angle = math.degrees(math.atan((y2 - y1) / (x2 - x1)))
        angles.append(angle)
    
    # Return the most common angle
    if angles:
        return np.median(angles)
    return 0

def rotate_image(image, angle):
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate the center of the image
    center = (width // 2, height // 2)
    
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new image dimensions
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)
    
    # Adjust the rotation matrix
    rotation_matrix[0, 2] += new_width / 2 - center[0]
    rotation_matrix[1, 2] += new_height / 2 - center[1]
    
    # Perform the rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    
    return rotated

def detect_card_region(image):
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to preserve edges while removing noise
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Try to find a rectangular contour
        for contour in contours[:5]:  # Check top 5 largest contours
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # If we have 4 points, we might have found our card
            if len(approx) == 4:
                # Get the minimum area rectangle
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype=np.int32)
                
                # Get the width and height of the rectangle
                width = rect[1][0]
                height = rect[1][1]
                
                # Ensure width is always greater than height
                if width < height:
                    width, height = height, width
                
                # Calculate aspect ratio
                aspect_ratio = width / height
                
                # Check if the contour is likely to be a credit card (aspect ratio should be around 1.586)
                if 1.4 < aspect_ratio < 1.7:
                    return box
        
        return None
    except Exception as e:
        print(f"Error in detect_card_region: {str(e)}")
        return None

def extract_number_region(image):
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on size and position
        valid_contours = []
        height, width = image.shape[:2]
        middle_third_y = height / 3
        
        # Sort contours by x-coordinate
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        # First pass: find potential digit contours
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)
            
            # More lenient criteria for digit detection
            if (0.1 < aspect_ratio < 1.2 and  # More lenient aspect ratio
                area > 50 and  # Smaller minimum area
                y > middle_third_y * 0.8 and  # More lenient vertical position
                y < middle_third_y * 2.2):
                
                # Check if this contour is too close to the previous one
                if valid_contours and x - valid_contours[-1][0] < 5:
                    continue
                    
                valid_contours.append((x, contour))
        
        # Second pass: if we don't have enough digits, try to split wide contours
        if len(valid_contours) < 16:
            new_contours = []
            for x, contour in valid_contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 1.5 * h:  # If contour is too wide, it might contain multiple digits
                    # Try to split into multiple digits
                    num_digits = max(2, min(4, int(w / h)))  # Estimate number of digits
                    for i in range(num_digits):
                        split_x = x + (w * i) // num_digits
                        new_contours.append((split_x, contour))
                else:
                    new_contours.append((x, contour))
            valid_contours = new_contours
        
        # Third pass: if we still don't have enough digits, try to find more contours
        if len(valid_contours) < 16:
            # Try with more lenient criteria
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                area = cv2.contourArea(contour)
                
                if (0.05 < aspect_ratio < 1.5 and  # Even more lenient aspect ratio
                    area > 30 and  # Even smaller minimum area
                    y > middle_third_y * 0.7 and  # More lenient vertical position
                    y < middle_third_y * 2.3):
                    
                    # Check if this contour is too close to any existing contour
                    too_close = False
                    for ex_x, _ in valid_contours:
                        if abs(x - ex_x) < 5:
                            too_close = True
                            break
                    
                    if not too_close:
                        valid_contours.append((x, contour))
        
        # Sort contours by x-coordinate
        valid_contours.sort(key=lambda x: x[0])
        
        # If we still don't have 16 digits, print warning
        if len(valid_contours) != 16:
            print(f"Warning: Found {len(valid_contours)} digits instead of 16")
        
        return valid_contours
    except Exception as e:
        print(f"Error in extract_number_region: {str(e)}")
        return []

def preprocess_digit(digit_image):
    try:
        # Ensure the image is binary
        _, digit = cv2.threshold(digit_image, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Add padding
        pad = 4
        digit = np.pad(digit, pad, mode='constant', constant_values=0)
        
        # Resize to 32x32
        digit = cv2.resize(digit, (32, 32))
        
        # Apply morphological operations to clean up the digit
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        digit = cv2.morphologyEx(digit, cv2.MORPH_CLOSE, kernel)
        
        return digit
    except Exception as e:
        print(f"Error in preprocess_digit: {str(e)}")
        return None

def extract_card_number(image_path, model):
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return "Error: Could not read image"
        
        # Resize image if it's too large
        max_dimension = 1000
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
        
        # Detect and correct card angle
        angle = detect_card_angle(image)
        rotated_image = rotate_image(image, angle)
        
        # Detect card region
        card_box = detect_card_region(rotated_image)
        if card_box is None:
            return "Error: Could not detect card"
        
        # Warp the card to a top-down view
        warped = four_point_transform(rotated_image, card_box)
        
        # Extract number region
        digit_contours = extract_number_region(warped)
        
        if not digit_contours:
            return "Error: Could not detect card numbers"
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        card_number = ""
        
        # Process each digit
        for _, contour in digit_contours:
            x, y, w, h = cv2.boundingRect(contour)
            digit = warped[y:y+h, x:x+w]
            
            # Convert to grayscale if not already
            if len(digit.shape) == 3:
                digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
            
            # Preprocess digit
            digit = preprocess_digit(digit)
            if digit is None:
                continue
            
            # Convert to tensor
            digit = torch.FloatTensor(digit).unsqueeze(0).unsqueeze(0) / 255.0
            digit = digit.to(device)
            
            # Predict digit
            with torch.no_grad():
                output = model(digit)
                predicted_digit = torch.argmax(output).item()
            card_number += str(predicted_digit)
        
        # Format the card number in 4-4-4-4 format
        if len(card_number) == 16:
            formatted_number = f"{card_number[:4]}-{card_number[4:8]}-{card_number[8:12]}-{card_number[12:]}"
            return formatted_number
        else:
            return f"Error: Expected 16 digits, got {len(card_number)}"
    
    except Exception as e:
        return f"Error processing card: {str(e)}"

if __name__ == "__main__":
    # Train the model
    model = train_model()
    
    # Example usage
    # Replace 'test_image.jpg' with your test image path
    card_number = extract_card_number('test_image.jpg', model)
    print(f"Extracted card number: {card_number}") 
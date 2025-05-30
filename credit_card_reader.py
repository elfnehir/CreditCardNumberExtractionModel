import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import time
import math
from typing import Tuple, List, Optional, Dict, Any
import logging
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('credit_card_ocr.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import yaml silently
YAML_AVAILABLE = False
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    pass

@dataclass
class ModelConfig:
    """Configuration for the model architecture"""
    input_channels: int = 1
    base_channels: int = 32
    num_classes: int = 10
    dropout_rate: float = 0.3
    use_attention: bool = True
    use_se: bool = True
    use_fpn: bool = True
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ModelConfig':
        """Load configuration from YAML file if available"""
        if not YAML_AVAILABLE:
            return cls()
        
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
        except Exception as e:
            logger.debug(f"Error loading YAML config: {str(e)}. Using default configuration.")
            return cls()

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AttentionBlock(nn.Module):
    """Self-attention block for spatial attention"""
    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        
        # Generate Q, K, V
        q = self.query(x).view(b, -1, h * w).permute(0, 2, 1)
        k = self.key(x).view(b, -1, h * w)
        v = self.value(x).view(b, -1, h * w)
        
        # Attention
        attention = torch.bmm(q, k)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        
        return self.gamma * out + x

class ResidualBlock(nn.Module):
    """Residual block with optional SE and attention"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 use_se: bool = True, use_attention: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.attention = AttentionBlock(out_channels) if use_attention else nn.Identity()
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.attention(out)
        out = self.dropout(out)
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class FPNBlock(nn.Module):
    """Feature Pyramid Network block for multi-scale feature fusion"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.upsample(x)
        x = x + skip
        x = self.conv2(x)
        return x

class AdvancedCreditCardNet(nn.Module):
    """Advanced CNN architecture with residual connections, attention, and FPN"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(config.input_channels, config.base_channels, 3, padding=1),
            nn.BatchNorm2d(config.base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(config.base_channels, config.base_channels, 2)
        self.layer2 = self._make_layer(config.base_channels, config.base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(config.base_channels * 2, config.base_channels * 4, 2, stride=2)
        
        # FPN
        if config.use_fpn:
            self.fpn1 = FPNBlock(config.base_channels * 4, config.base_channels * 2)
            self.fpn2 = FPNBlock(config.base_channels * 2, config.base_channels)
        
        # Calculate the final feature map size
        self.final_channels = config.base_channels * 4 if not config.use_fpn else config.base_channels
        
        # Classification head with proper dimensions
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # This will output (batch_size, final_channels, 1, 1)
            nn.Flatten(),  # This will output (batch_size, final_channels)
            nn.Linear(self.final_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(512, config.num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int,
                   stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride,
                                  self.config.use_se, self.config.use_attention))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels,
                                      use_se=self.config.use_se,
                                      use_attention=self.config.use_attention))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial features
        x = self.conv1(x)
        
        # Residual blocks
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        
        # FPN
        if self.config.use_fpn:
            p3 = c3
            p2 = self.fpn1(p3, c2)
            p1 = self.fpn2(p2, c1)
            x = p1
        else:
            x = c3
        
        # Classification
        x = self.classifier(x)
        return x

class AdvancedImageProcessor:
    """Enhanced image processing pipeline with multiple methods"""
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing pipeline with multiple methods"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply multiple preprocessing methods
            processed_images = []
            
            # Method 1: Bilateral filter + CLAHE
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            enhanced1 = self.clahe.apply(filtered)
            processed_images.append(enhanced1)
            
            # Method 2: Gaussian blur + Adaptive threshold
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            enhanced2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
            processed_images.append(enhanced2)
            
            # Method 3: Otsu's thresholding
            _, enhanced3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(enhanced3)
            
            # Combine results
            final = np.zeros_like(gray)
            for img in processed_images:
                final = cv2.bitwise_or(final, img)
            
            # Apply unsharp masking for edge enhancement
            gaussian = cv2.GaussianBlur(final, (0, 0), 3.0)
            final = cv2.addWeighted(final, 1.5, gaussian, -0.5, 0)
            
            # Normalize
            final = cv2.normalize(final, None, 0, 255, cv2.NORM_MINMAX)
            
            return final
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            return image

    def detect_card(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Enhanced card detection with multiple methods"""
        try:
            # Try multiple preprocessing methods
            processed_images = []
            
            # Method 1: Original preprocessing
            processed_images.append(self.preprocess(image))
            
            # Method 2: Canny edge detection
            edges = cv2.Canny(image, 50, 150)
            processed_images.append(edges)
            
            # Method 3: Sobel edges
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobelx**2 + sobely**2)
            sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            processed_images.append(sobel)
            
            for processed in processed_images:
                # Find contours
                contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    continue
                
                # Sort contours by area
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                for contour in contours[:5]:
                    # Try multiple approximation methods
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                    
                    if len(approx) == 4:
                        rect = cv2.minAreaRect(contour)
                        box = cv2.boxPoints(rect)
                        box = np.array(box, dtype=np.int32)
                        
                        # Check aspect ratio
                        width = rect[1][0]
                        height = rect[1][1]
                        if width < height:
                            width, height = height, width
                        aspect_ratio = width / height
                        
                        if 1.4 < aspect_ratio < 1.7:
                            return box
            
            return None
        except Exception as e:
            logger.error(f"Error in card detection: {str(e)}")
            return None

    def extract_digits(self, image: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """Enhanced digit extraction with multiple passes and validation"""
        try:
            # Ensure image is grayscale and uint8
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            if gray.dtype != np.uint8:
                gray = (gray * 255).astype(np.uint8)
            
            # Apply multiple thresholding methods
            thresh_methods = [
                lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2),
                lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
                lambda img: cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1]
            ]
            
            valid_contours = []
            height, width = gray.shape[:2]
            middle_third_y = height / 3
            
            for thresh_method in thresh_methods:
                try:
                    # Apply thresholding
                    thresh = thresh_method(gray)
                    
                    # Apply morphological operations
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                    
                    # Find contours
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Filter contours
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / float(h)
                        area = cv2.contourArea(contour)
                        
                        if (0.1 < aspect_ratio < 1.2 and
                            area > 50 and
                            y > middle_third_y * 0.8 and
                            y < middle_third_y * 2.2):
                            
                            # Check if this contour is too close to any existing contour
                            too_close = False
                            for ex_x, _ in valid_contours:
                                if abs(x - ex_x) < 5:
                                    too_close = True
                                    break
                            
                            if not too_close:
                                valid_contours.append((x, contour))
                    
                    # If we found enough digits, break
                    if len(valid_contours) >= 16:
                        break
                except Exception as e:
                    logger.warning(f"Error in threshold method: {str(e)}")
                    continue
            
            # Sort contours by x-coordinate
            valid_contours.sort(key=lambda x: x[0])
            
            # If we still don't have 16 digits, try to split wide contours
            if len(valid_contours) < 16:
                new_contours = []
                for x, contour in valid_contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 1.5 * h:
                        # Try to split into multiple digits
                        num_digits = max(2, min(4, int(w / h)))
                        for i in range(num_digits):
                            split_x = x + (w * i) // num_digits
                            new_contours.append((split_x, contour))
                    else:
                        new_contours.append((x, contour))
                valid_contours = new_contours
            
            if len(valid_contours) != 16:
                logger.warning(f"Found {len(valid_contours)} digits instead of 16")
            
            return valid_contours
        except Exception as e:
            logger.error(f"Error in digit extraction: {str(e)}")
            return []

class CreditCardDataset(Dataset):
    """Dataset class for credit card digits"""
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform: Optional[transforms.Compose] = None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_and_prepare_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and prepare the dataset for training"""
    try:
        images = []
        labels = []
        
        # Load images from each digit folder
        for digit in range(10):
            digit_path = os.path.join(dataset_path, str(digit))
            if not os.path.exists(digit_path):
                logger.warning(f"Directory {digit_path} does not exist")
                continue
                
            for img_name in os.listdir(digit_path):
                img_path = os.path.join(digit_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize to 32x32 and convert to grayscale
                    img = cv2.resize(img, (32, 32))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    images.append(img)
                    labels.append(digit)
        
        if not images:
            raise ValueError("No images found in the dataset")
        
        return np.array(images), np.array(labels)
    except Exception as e:
        logger.error(f"Error in load_and_prepare_dataset: {str(e)}")
        raise

def train_model(config: ModelConfig) -> nn.Module:
    """Enhanced model training with better optimization"""
    try:
        # Load and prepare dataset
        X, y = load_and_prepare_dataset('Credit Card Number Dataset')
        
        # Convert to PyTorch tensors and ensure correct shape
        X = torch.FloatTensor(X).unsqueeze(1) / 255.0  # Add channel dimension
        y = torch.LongTensor(y)
        
        # Print shapes for debugging
        logger.info(f"Input tensor shape: {X.shape}")
        
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
        
        # Create data loaders with reduced number of workers
        batch_size = 32  # Reduced batch size for better stability
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)
        
        # Create and train model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AdvancedCreditCardNet(config).to(device)
        
        # Print model summary
        logger.info(f"Model architecture:\n{model}")
        
        # Use AdamW optimizer with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Use CosineAnnealingWarmRestarts for better learning rate scheduling
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        criterion = nn.CrossEntropyLoss()
        
        # Training parameters
        num_epochs = 100
        best_accuracy = 0
        patience = 20
        patience_counter = 0
        min_epochs = 50
        
        # Training loop with progress bar
        for epoch in range(num_epochs):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Training phase with progress bar
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for images, labels in train_pbar:
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
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
            
            train_accuracy = 100 * correct / total
            train_loss = running_loss / len(train_loader)
            
            # Validation phase with progress bar
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0
            
            val_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            with torch.no_grad():
                for images, labels in val_pbar:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    # Update progress bar
                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100 * val_correct / val_total:.2f}%'
                    })
            
            val_accuracy = 100 * val_correct / val_total
            val_loss = val_loss / len(test_loader)
            
            # Update learning rate
            scheduler.step()
            
            # Early stopping with minimum epochs
            if epoch >= min_epochs:
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
            
            logger.info(f'Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.2f}s)')
            logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            
            if epoch >= min_epochs and patience_counter >= patience:
                logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
        
        # Load best model
        checkpoint = torch.load('best_credit_card_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        raise

def format_card_number(digits: List[str]) -> str:
    """Format card number with proper spacing and error handling"""
    try:
        # Initialize result with all X's
        result = ['xxxx', 'xxxx', 'xxxx', 'xxxx']
        
        # Fill in detected digits
        for i, digit in enumerate(digits):
            group_idx = i // 4
            pos_in_group = i % 4
            if group_idx < 4:  # Ensure we don't go beyond 4 groups
                if digit != 'x':
                    result[group_idx] = result[group_idx][:pos_in_group] + digit + result[group_idx][pos_in_group + 1:]
        
        # Join groups with spaces
        return ' '.join(result)
    except Exception as e:
        logger.error(f"Error in formatting card number: {str(e)}")
        return 'xxxx xxxx xxxx xxxx'

def extract_card_number(image_path: str, model: nn.Module) -> str:
    """Enhanced card number extraction with better error handling and reporting"""
    try:
        # Initialize image processor
        processor = AdvancedImageProcessor()
        
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
        
        # Try multiple preprocessing methods
        processed_images = []
        processed_images.append(processor.preprocess(image))
        processed_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        
        card_number = None
        max_confidence = 0
        
        for processed in processed_images:
            # Detect card region
            card_box = processor.detect_card(processed)
            if card_box is None:
                continue
            
            # Warp the card to a top-down view
            warped = four_point_transform(processed, card_box)
            
            # Extract number region
            digit_contours = processor.extract_digits(warped)
            
            if not digit_contours:
                continue
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
            
            # Process each digit
            digits = []
            total_confidence = 0
            
            for _, contour in digit_contours:
                x, y, w, h = cv2.boundingRect(contour)
                digit = warped[y:y+h, x:x+w]
                
                # Convert to grayscale if not already
                if len(digit.shape) == 3:
                    digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
                
                # Preprocess digit
                digit = preprocess_digit(digit)
                if digit is None:
                    digits.append('x')
                    continue
                
                # Convert to tensor
                digit = torch.FloatTensor(digit).unsqueeze(0).unsqueeze(0) / 255.0
                digit = digit.to(device)
                
                # Predict digit
                with torch.no_grad():
                    output = model(digit)
                    probabilities = F.softmax(output, dim=1)
                    confidence, predicted_digit = torch.max(probabilities, 1)
                    total_confidence += confidence.item()
                    digits.append(str(predicted_digit.item()))
            
            # Calculate average confidence
            avg_confidence = total_confidence / len(digits)
            
            # If this attempt has higher confidence, update the result
            if avg_confidence > max_confidence:
                max_confidence = avg_confidence
                card_number = format_card_number(digits)
        
        if card_number is None:
            return "Error: Could not detect card number"
        
        return card_number
    
    except Exception as e:
        logger.error(f"Error in extract_card_number: {str(e)}")
        return f"Error processing card: {str(e)}"

def main():
    """Main function to handle the program execution"""
    try:
        # Load configuration
        config = ModelConfig()
        
        # Train the model
        logger.info("Starting model training...")
        model = train_model(config)
        logger.info("Model training completed")
        
        # Process test images
        test_images = [f for f in os.listdir('.') if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for image_path in test_images:
            logger.info(f"\nProcessing image: {image_path}")
            card_number = extract_card_number(image_path, model)
            logger.info(f"Detected card number: {card_number}")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
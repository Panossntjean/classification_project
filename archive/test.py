import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

# 1. Define paths
train_path = "data/train/"
sunny_path = os.path.join(train_path, "sunny")
cloud_path = os.path.join(train_path, "cloud")

# 2. Image analysis function using PIL
def analyze_images(folder_path):
    sizes = []
    means = []
    stds = []
    red_values = []
    green_values = []
    blue_values = []
    
    for img_file in os.listdir(folder_path):
        if img_file.endswith('.jpg'):
            img_path = os.path.join(folder_path, img_file)
            with Image.open(img_path) as img:
                img_array = np.array(img)
                
                # Basic image characteristics
                sizes.append(img_array.shape)
                
                # Color channels (if RGB)
                if len(img_array.shape) == 3:  # Color image
                    red = img_array[:,:,0].flatten()
                    green = img_array[:,:,1].flatten()
                    blue = img_array[:,:,2].flatten()
                else:  # Grayscale
                    red = green = blue = img_array.flatten()
                
                red_values.extend(red)
                green_values.extend(green)
                blue_values.extend(blue)
                
                # Mean and standard deviation
                means.append(np.mean(img_array))
                stds.append(np.std(img_array))
    
    return {
        'sizes': sizes,
        'means': means,
        'stds': stds,
        'red_values': red_values,
        'green_values': green_values,
        'blue_values': blue_values
    }

# Analyze both classes
sunny_stats = analyze_images(sunny_path)
cloud_stats = analyze_images(cloud_path)

# 3. Statistical description
def print_stats(stats, class_name):
    print(f"\nStatistics for class: {class_name}")
    print(f"Number of images: {len(stats['means'])}")
    print(f"Mean brightness: {np.mean(stats['means']):.2f}")
    print(f"Median brightness: {np.median(stats['means']):.2f}")
    print(f"Brightness std dev: {np.mean(stats['stds']):.2f}")
    print(f"Min brightness: {np.min(stats['means']):.2f}")
    print(f"Max brightness: {np.max(stats['means']):.2f}")
    
    # Image dimensions analysis
    sizes = np.array(stats['sizes'])
    if len(sizes) > 0:
        print("\nImage dimensions (height, width, channels):")
        print(f"Mean: {np.mean(sizes, axis=0)}")
        print(f"Min: {np.min(sizes, axis=0)}")
        print(f"Max: {np.max(sizes, axis=0)}")

print_stats(sunny_stats, "sunny")
print_stats(cloud_stats, "cloud")

# 4. Class distribution analysis
def plot_class_distribution():
    sunny_count = len([f for f in os.listdir(sunny_path) if f.endswith('.jpg')])
    cloud_count = len([f for f in os.listdir(cloud_path) if f.endswith('.jpg')])
    
    plt.figure(figsize=(8, 6))
    plt.bar(['Sunny', 'Cloud'], [sunny_count, cloud_count], color=['gold', 'gray'])
    plt.title('Class Distribution')
    plt.ylabel('Number of Images')
    plt.show()
    
    # Calculate percentages
    total = sunny_count + cloud_count
    print(f"\nClass balance analysis:")
    print(f"Sunny: {sunny_count} images ({sunny_count/total*100:.2f}%)")
    print(f"Cloud: {cloud_count} images ({cloud_count/total*100:.2f}%)")
    
    # Check for class imbalance (typically considered imbalanced when ratio > 2:1)
    if max(sunny_count, cloud_count) / min(sunny_count, cloud_count) > 2:
        print("\nWARNING: Significant class imbalance detected!")
    else:
        print("\nClasses are relatively balanced.")

plot_class_distribution()

# 5. Color channel analysis and outliers
def analyze_color_channels():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Color Channel Distribution by Class')
    
    # Sunny class
    axes[0,0].hist(sunny_stats['red_values'], bins=50, color='red', alpha=0.7)
    axes[0,0].set_title('Sunny - Red Channel')
    
    axes[0,1].hist(sunny_stats['green_values'], bins=50, color='green', alpha=0.7)
    axes[0,1].set_title('Sunny - Green Channel')
    
    axes[0,2].hist(sunny_stats['blue_values'], bins=50, color='blue', alpha=0.7)
    axes[0,2].set_title('Sunny - Blue Channel')
    
    # Cloud class
    axes[1,0].hist(cloud_stats['red_values'], bins=50, color='red', alpha=0.7)
    axes[1,0].set_title('Cloud - Red Channel')
    
    axes[1,1].hist(cloud_stats['green_values'], bins=50, color='green', alpha=0.7)
    axes[1,1].set_title('Cloud - Green Channel')
    
    axes[1,2].hist(cloud_stats['blue_values'], bins=50, color='blue', alpha=0.7)
    axes[1,2].set_title('Cloud - Blue Channel')
    
    plt.tight_layout()
    plt.show()
    
    # Check for outliers in brightness means
    plt.figure(figsize=(10, 6))
    plt.boxplot([sunny_stats['means'], cloud_stats['means']], labels=['Sunny', 'Cloud'])
    plt.title('Range of Mean Brightness Values by Class')
    plt.ylabel('Mean pixel value')
    plt.show()
    
    # Identify outliers using IQR rule
    def detect_outliers(data):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        return outliers
    
    sunny_outliers = detect_outliers(sunny_stats['means'])
    cloud_outliers = detect_outliers(cloud_stats['means'])
    
    print(f"\nOutliers in mean brightness:")
    print(f"Sunny: {len(sunny_outliers)} outliers ({len(sunny_outliers)/len(sunny_stats['means'])*100:.2f}%)")
    print(f"Cloud: {len(cloud_outliers)} outliers ({len(cloud_outliers)/len(cloud_stats['means'])*100:.2f}%)")

analyze_color_channels()

# 6. Approach for handling outliers
print("\nApproach for handling outliers:")
print("1. Evaluation: Examine images corresponding to outliers for potential errors.")
print("2. Retention: If outliers represent genuine extreme cases (e.g., very dark sunny images), they might be useful for the model.")
print("3. Removal: If outliers result from errors (e.g., corrupted images), they can be removed.")
print("4. Normalization: Apply normalization or standard scaling to reduce the impact of outliers.")
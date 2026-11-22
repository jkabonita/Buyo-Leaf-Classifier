# 🍃 Buyo Leaf Classification Guide

## Overview
This document defines the classification criteria for Buyo (Betel) leaves used in the automated sorting system.

---

## 📊 Classification Classes

### **CLASS A - Premium Quality**
**Characteristics:**
- **Color**: Deep green (RGB: 34-100, 139-180, 34-100)
- **Size**: Large (>15cm length)
- **Shape**: Perfect heart-shaped, symmetrical
- **Texture**: Smooth, glossy surface
- **Defects**: None or minimal (<2% surface area)
- **Veins**: Prominent, well-defined
- **Age**: Young to mature leaves

**Quality Score**: 90-100%
**Market Value**: Premium/Export grade
**Use Cases**: Religious ceremonies, premium paan

---

### **CLASS B - High Quality**
**Characteristics:**
- **Color**: Medium to dark green (RGB: 50-120, 130-160, 50-110)
- **Size**: Medium to large (12-15cm length)
- **Shape**: Good heart shape, minor asymmetry acceptable
- **Texture**: Smooth with slight variations
- **Defects**: Minor (2-5% surface area)
  - Small spots
  - Slight discoloration
  - Minor edge damage
- **Veins**: Clear and visible
- **Age**: Mature leaves

**Quality Score**: 75-89%
**Market Value**: High grade
**Use Cases**: Commercial paan, restaurant supply

---

### **CLASS C - Standard Quality**
**Characteristics:**
- **Color**: Light to medium green (RGB: 70-140, 120-150, 70-120)
- **Size**: Medium (10-12cm length)
- **Shape**: Heart-shaped but irregular
- **Texture**: Slightly rough or uneven
- **Defects**: Moderate (5-15% surface area)
  - Multiple small spots
  - Visible discoloration patches
  - Edge damage or tears
  - Minor holes (<5mm)
- **Veins**: Visible but less prominent
- **Age**: Mature to older leaves

**Quality Score**: 50-74%
**Market Value**: Standard grade
**Use Cases**: Local market, general consumption

---

### **CLASS D - Below Standard**
**Characteristics:**
- **Color**: Pale green or yellowish (RGB: 100-160, 150-180, 80-130)
- **Size**: Small to medium (8-10cm length)
- **Shape**: Irregular, deformed
- **Texture**: Rough, wilted appearance
- **Defects**: Significant (15-30% surface area)
  - Large spots or patches
  - Yellowing areas
  - Multiple tears
  - Holes (5-10mm)
  - Insect damage visible
- **Veins**: Faded or discolored
- **Age**: Old or damaged leaves

**Quality Score**: 25-49%
**Market Value**: Low grade
**Use Cases**: Processing, extract production

---

### **CLASS E - Reject/Defective**
**Characteristics:**
- **Color**: Yellow, brown, or severely discolored
- **Size**: Any size
- **Shape**: Severely deformed or broken
- **Texture**: Wilted, dried, or decaying
- **Defects**: Severe (>30% surface area)
  - Large brown/black spots
  - Extensive yellowing
  - Major tears or missing sections
  - Large holes (>10mm)
  - Fungal infection
  - Pest damage
  - Decay or rot
- **Veins**: Brown or black discoloration
- **Age**: Very old, dying, or dead leaves

**Quality Score**: 0-24%
**Market Value**: Reject/Discard
**Use Cases**: Compost, not suitable for consumption

---

## 🎨 Color Analysis Parameters

### **Healthy Green Range (HSV)**
- **Hue**: 80-140° (green spectrum)
- **Saturation**: 30-80% (vibrant but not oversaturated)
- **Value/Brightness**: 40-80% (not too dark, not washed out)

### **Defect Color Indicators**
- **Yellow Spots**: H: 40-70°, S: >40%
- **Brown Spots**: H: 15-35°, S: >30%, V: <50%
- **Black Spots**: V: <20%
- **White/Bleached**: S: <20%, V: >80%

---

## 🔍 Defect Detection Categories

### **1. Color Defects**
- **Yellowing**: Chlorophyll degradation, nutrient deficiency
- **Brown Spots**: Disease, aging, oxidation
- **White Patches**: Powdery mildew, sun damage
- **Black Spots**: Fungal infection, bacterial disease

### **2. Physical Defects**
- **Holes**: Insect damage, mechanical damage
- **Tears**: Handling damage, wind damage
- **Missing Sections**: Severe physical damage
- **Deformation**: Growth abnormalities

### **3. Texture Defects**
- **Wilting**: Water stress, age
- **Wrinkles**: Dehydration, over-maturity
- **Bumps/Galls**: Insect larvae, disease
- **Rough Surface**: Fungal growth

### **4. Edge Defects**
- **Brown Edges**: Tip burn, water stress
- **Ragged Edges**: Mechanical damage, disease
- **Curled Edges**: Environmental stress

---

## 📐 Size Measurement Standards

### **Length Classification**
- **Extra Large**: >15cm
- **Large**: 12-15cm
- **Medium**: 10-12cm
- **Small**: 8-10cm
- **Very Small**: <8cm

### **Width Classification**
- **Wide**: >10cm (at widest point)
- **Medium**: 8-10cm
- **Narrow**: <8cm

### **Area Calculation**
Approximate area = Length × Width × 0.65 (heart shape factor)

---

## 🔬 Detection Algorithms

### **Color Analysis Pipeline**
1. Convert BGR → HSV color space
2. Extract green channel histogram
3. Calculate mean, std, and distribution
4. Identify non-green regions (defects)
5. Calculate defect percentage

### **Defect Detection Pipeline**
1. Apply color thresholding for defect colors
2. Use morphological operations (erosion/dilation)
3. Contour detection for spot/hole identification
4. Calculate total defect area
5. Classify severity based on percentage

### **Shape Analysis Pipeline**
1. Edge detection (Canny)
2. Contour extraction
3. Calculate symmetry score
4. Measure aspect ratio
5. Detect deformations

---

## 🎯 Classification Algorithm

### **Scoring System (0-100)**

**Color Health Score (40%)**
- Green intensity: 15 points
- Color uniformity: 15 points
- Hue consistency: 10 points

**Defect Score (35%)**
- Surface defects: 20 points
- Edge defects: 10 points
- Texture defects: 5 points

**Shape Score (15%)**
- Symmetry: 8 points
- Heart shape accuracy: 7 points

**Size Score (10%)**
- Length: 5 points
- Width: 5 points

### **Final Classification**
- **CLASS A**: Score ≥ 90, defects < 2%
- **CLASS B**: Score 75-89, defects 2-5%
- **CLASS C**: Score 50-74, defects 5-15%
- **CLASS D**: Score 25-49, defects 15-30%
- **CLASS E**: Score < 25, defects > 30%

---

## 🛠️ Python Libraries Required

### **Core Libraries**
```python
opencv-python>=4.5.0       # Image processing and analysis
numpy>=1.20.0              # Numerical operations
scipy>=1.7.0               # Scientific computing
scikit-image>=0.18.0       # Advanced image processing
```

### **Color Analysis**
```python
colormath>=3.0.0           # Color space conversions
colorthief>=0.2.1          # Dominant color extraction
```

### **Machine Learning**
```python
torch>=2.0.0               # PyTorch for neural networks
torchvision>=0.15.0        # Computer vision models
```

### **Visualization**
```python
matplotlib>=3.3.0          # Plotting and visualization
seaborn>=0.11.0            # Statistical visualization
```

---

## 📊 Quality Metrics Display

### **GUI Display Format**
```
╔═══════════════════════════════════════╗
║  🍃 BUYO LEAF ANALYSIS               ║
╠═══════════════════════════════════════╣
║  Class:        CLASS B                ║
║  Confidence:   85.3%                  ║
║  Quality:      82/100 ⭐⭐⭐⭐        ║
╠═══════════════════════════════════════╣
║  📊 DETAILED ANALYSIS                 ║
║  Color Health:   36/40  [████████▒▒] ║
║  Defects:        28/35  [███████▒▒▒] ║
║  Shape:          12/15  [████████▒▒] ║
║  Size:           6/10   [██████▒▒▒▒] ║
╠═══════════════════════════════════════╣
║  🎨 COLOR PROFILE                     ║
║  Primary:    Dark Green (#2d7a4f)    ║
║  Uniformity: 78%                      ║
║  HSV:        H:135° S:65% V:48%      ║
╠═══════════════════════════════════════╣
║  ⚠️ DEFECTS DETECTED                  ║
║  • Small spots (3): 2.1% area         ║
║  • Edge damage: 1.8% area             ║
║  • Total defect: 3.9% ✓               ║
╠═══════════════════════════════════════╣
║  📐 MEASUREMENTS                       ║
║  Length:  13.2 cm                     ║
║  Width:   9.8 cm                      ║
║  Area:    84.2 cm²                    ║
╚═══════════════════════════════════════╝
```

---

## 🔧 Calibration Guidelines

### **Camera Setup**
- **Distance**: 20-30 cm from leaf
- **Lighting**: Diffused white light, 500-1000 lux
- **Background**: White or neutral (for contrast)
- **Angle**: 90° perpendicular to leaf surface

### **Color Calibration**
1. Use ColorChecker card for white balance
2. Capture reference green sample (known CLASS A leaf)
3. Adjust RGB/HSV thresholds based on environment
4. Test with known samples of each class

### **Size Calibration**
1. Place ruler in frame for scale reference
2. Capture calibration image
3. Calculate pixels-to-cm conversion factor
4. Apply scale factor to all measurements

---

## 📝 Notes

- Classification criteria may vary by region and market standards
- Seasonal variations affect color ranges (adjust thresholds accordingly)
- Fresh leaves have higher moisture and deeper green color
- Storage conditions affect leaf quality over time
- Regular calibration recommended (weekly) for consistent results

---

**Document Version**: 1.0
**Last Updated**: 2024
**Author**: Buyo Leaf Classification System

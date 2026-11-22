#!/usr/bin/env python3
"""
Buyo Leaf Analyzer - Advanced Color and Defect Detection
=========================================================
Analyzes Buyo leaves for:
- Color health and uniformity
- Defect detection (spots, holes, discoloration)
- Shape and size measurements
- Quality scoring
"""

import cv2
import numpy as np
from scipy import ndimage
from typing import Dict, Tuple, List

class BuyoLeafAnalyzer:
    """Advanced Buyo leaf analysis with color and defect detection"""
    
    # Color thresholds in HSV
    HEALTHY_GREEN_HSV = {
        'h_min': 80, 'h_max': 140,  # Hue (green range)
        's_min': 30, 's_max': 80,   # Saturation
        'v_min': 40, 'v_max': 80    # Value/Brightness
    }
    
    # Defect color thresholds
    DEFECT_COLORS = {
        'yellow': {'h_min': 40, 'h_max': 70, 's_min': 40},
        'brown': {'h_min': 15, 'h_max': 35, 's_min': 30, 'v_max': 50},
        'black': {'v_max': 20},
        'white': {'s_max': 20, 'v_min': 80}
    }
    
    # Classification thresholds
    CLASS_THRESHOLDS = {
        'A': {'score_min': 90, 'defect_max': 2.0},
        'B': {'score_min': 75, 'defect_max': 5.0},
        'C': {'score_min': 50, 'defect_max': 15.0},
        'D': {'score_min': 25, 'defect_max': 30.0},
        'E': {'score_min': 0, 'defect_max': 100.0}
    }
    
    def __init__(self):
        """Initialize the analyzer"""
        pass
    
    def analyze_leaf(self, image: np.ndarray) -> Dict:
        """
        Perform comprehensive leaf analysis
        
        Args:
            image: BGR image of the leaf
            
        Returns:
            Dictionary with analysis results
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Analyze color health
        color_analysis = self._analyze_color(image, hsv)
        
        # Detect defects
        defect_analysis = self._detect_defects(image, hsv)
        
        # Analyze shape
        shape_analysis = self._analyze_shape(image)
        
        # Calculate scores
        scores = self._calculate_scores(color_analysis, defect_analysis, shape_analysis)
        
        # Determine quality grade
        quality_grade = self._determine_quality_grade(scores, defect_analysis['total_defect_percent'])
        
        return {
            'color': color_analysis,
            'defects': defect_analysis,
            'shape': shape_analysis,
            'scores': scores,
            'quality_grade': quality_grade,
            'overall_score': scores['total'],
            'quality_stars': self._get_quality_stars(scores['total'])
        }
    
    def _analyze_color(self, image: np.ndarray, hsv: np.ndarray) -> Dict:
        """Analyze color health and uniformity"""
        h, s, v = cv2.split(hsv)
        
        # Calculate green mask
        green_mask = cv2.inRange(hsv, 
                                 np.array([self.HEALTHY_GREEN_HSV['h_min'], 
                                          self.HEALTHY_GREEN_HSV['s_min'],
                                          self.HEALTHY_GREEN_HSV['v_min']]),
                                 np.array([self.HEALTHY_GREEN_HSV['h_max'],
                                          self.HEALTHY_GREEN_HSV['s_max'],
                                          self.HEALTHY_GREEN_HSV['v_max']]))
        
        green_pixels = np.sum(green_mask > 0)
        total_pixels = image.shape[0] * image.shape[1]
        green_ratio = (green_pixels / total_pixels) * 100
        
        # Calculate color statistics
        mean_h = np.mean(h)
        mean_s = np.mean(s)
        mean_v = np.mean(v)
        
        std_h = np.std(h)
        
        # Color uniformity (lower std = more uniform)
        uniformity = max(0, 100 - (std_h / 180 * 100))
        
        # Extract dominant color
        dominant_color = self._get_dominant_color(image)
        
        return {
            'green_ratio': green_ratio,
            'uniformity': uniformity,
            'mean_hue': mean_h,
            'mean_saturation': mean_s,
            'mean_value': mean_v,
            'std_hue': std_h,
            'dominant_color_rgb': dominant_color,
            'dominant_color_hex': self._rgb_to_hex(dominant_color)
        }
    
    def _detect_defects(self, image: np.ndarray, hsv: np.ndarray) -> Dict:
        """Detect various types of defects"""
        defects = []
        total_defect_area = 0
        total_pixels = image.shape[0] * image.shape[1]
        
        # Detect yellow spots (chlorosis)
        yellow_mask = cv2.inRange(hsv,
                                  np.array([self.DEFECT_COLORS['yellow']['h_min'],
                                           self.DEFECT_COLORS['yellow']['s_min'], 0]),
                                  np.array([self.DEFECT_COLORS['yellow']['h_max'], 255, 255]))
        yellow_area = np.sum(yellow_mask > 0)
        if yellow_area > total_pixels * 0.01:  # >1% area
            defects.append({
                'type': 'Yellow spots',
                'area_pixels': int(yellow_area),
                'area_percent': (yellow_area / total_pixels) * 100
            })
            total_defect_area += yellow_area
        
        # Detect brown spots (disease/aging)
        brown_mask = cv2.inRange(hsv,
                                np.array([self.DEFECT_COLORS['brown']['h_min'],
                                         self.DEFECT_COLORS['brown']['s_min'], 0]),
                                np.array([self.DEFECT_COLORS['brown']['h_max'],
                                         255, self.DEFECT_COLORS['brown']['v_max']]))
        brown_area = np.sum(brown_mask > 0)
        if brown_area > total_pixels * 0.01:
            defects.append({
                'type': 'Brown spots',
                'area_pixels': int(brown_area),
                'area_percent': (brown_area / total_pixels) * 100
            })
            total_defect_area += brown_area
        
        # Detect black spots (severe disease)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        black_mask = cv2.threshold(gray, self.DEFECT_COLORS['black']['v_max'], 255, cv2.THRESH_BINARY_INV)[1]
        black_area = np.sum(black_mask > 0)
        if black_area > total_pixels * 0.005:  # >0.5% area
            defects.append({
                'type': 'Black spots',
                'area_pixels': int(black_area),
                'area_percent': (black_area / total_pixels) * 100
            })
            total_defect_area += black_area
        
        # Detect holes (edge detection + contour analysis)
        holes = self._detect_holes(image)
        if holes['count'] > 0:
            defects.append({
                'type': f'Holes ({holes["count"]})',
                'area_pixels': holes['total_area'],
                'area_percent': (holes['total_area'] / total_pixels) * 100
            })
            total_defect_area += holes['total_area']
        
        total_defect_percent = (total_defect_area / total_pixels) * 100
        
        return {
            'defects': defects,
            'total_defect_area': int(total_defect_area),
            'total_defect_percent': total_defect_percent,
            'defect_count': len(defects)
        }
    
    def _detect_holes(self, image: np.ndarray) -> Dict:
        """Detect holes in the leaf"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to find dark regions
        _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hole_count = 0
        total_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter small noise (< 50 pixels) and very large regions (> 5% of image)
            if 50 < area < (image.shape[0] * image.shape[1] * 0.05):
                hole_count += 1
                total_area += area
        
        return {
            'count': hole_count,
            'total_area': int(total_area)
        }
    
    def _analyze_shape(self, image: np.ndarray) -> Dict:
        """Analyze leaf shape and size"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find leaf contour
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'error': 'No contour found'}
        
        # Get largest contour (assume it's the leaf)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Calculate area
        area = cv2.contourArea(main_contour)
        
        # Calculate aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Calculate symmetry (compare left and right halves)
        symmetry = self._calculate_symmetry(main_contour, x, y, w, h)
        
        return {
            'width_px': w,
            'height_px': h,
            'area_px': int(area),
            'aspect_ratio': aspect_ratio,
            'symmetry_score': symmetry
        }
    
    def _calculate_symmetry(self, contour, x, y, w, h) -> float:
        """Calculate left-right symmetry score"""
        try:
            # Create mask for contour
            mask = np.zeros((h, w), dtype=np.uint8)
            shifted_contour = contour - [x, y]
            cv2.drawContours(mask, [shifted_contour], -1, 255, -1)
            
            # Split into left and right halves
            mid = w // 2
            left_half = mask[:, :mid]
            right_half = cv2.flip(mask[:, mid:], 1)
            
            # Resize to same size if needed
            if left_half.shape[1] != right_half.shape[1]:
                min_width = min(left_half.shape[1], right_half.shape[1])
                left_half = left_half[:, :min_width]
                right_half = right_half[:, :min_width]
            
            # Calculate similarity
            intersection = np.sum((left_half > 0) & (right_half > 0))
            union = np.sum((left_half > 0) | (right_half > 0))
            
            symmetry = (intersection / union * 100) if union > 0 else 0
            return symmetry
        except:
            return 50.0  # Default moderate symmetry
    
    def _calculate_scores(self, color_analysis: Dict, defect_analysis: Dict, shape_analysis: Dict) -> Dict:
        """Calculate quality scores"""
        # Color health score (40 points max)
        color_score = 0
        if color_analysis['green_ratio'] > 60:
            color_score += 15  # Green intensity
        elif color_analysis['green_ratio'] > 40:
            color_score += 10
        elif color_analysis['green_ratio'] > 20:
            color_score += 5
        
        color_score += min(15, color_analysis['uniformity'] * 0.15)  # Uniformity
        
        # Hue consistency (within healthy range)
        if self.HEALTHY_GREEN_HSV['h_min'] <= color_analysis['mean_hue'] <= self.HEALTHY_GREEN_HSV['h_max']:
            color_score += 10
        
        # Defect score (35 points max)
        defect_penalty = defect_analysis['total_defect_percent']
        defect_score = max(0, 35 - (defect_penalty * 1.5))  # Reduce 1.5 points per 1% defect
        
        # Shape score (15 points max)
        shape_score = 0
        if 'symmetry_score' in shape_analysis:
            shape_score += min(8, shape_analysis['symmetry_score'] * 0.08)  # Symmetry
            # Aspect ratio (heart shape is ~0.7-0.8)
            if 'aspect_ratio' in shape_analysis:
                aspect_diff = abs(shape_analysis['aspect_ratio'] - 0.75)
                shape_score += max(0, 7 - (aspect_diff * 10))
        
        # Size score (10 points max)
        size_score = 5  # Default medium size score
        if 'area_px' in shape_analysis:
            if shape_analysis['area_px'] > 100000:
                size_score = 10  # Large
            elif shape_analysis['area_px'] > 50000:
                size_score = 7  # Medium
            else:
                size_score = 3  # Small
        
        total_score = color_score + defect_score + shape_score + size_score
        
        return {
            'color': round(color_score, 1),
            'defect': round(defect_score, 1),
            'shape': round(shape_score, 1),
            'size': round(size_score, 1),
            'total': round(total_score, 1)
        }
    
    def _determine_quality_grade(self, scores: Dict, defect_percent: float) -> str:
        """Determine quality grade based on score and defects"""
        total_score = scores['total']
        
        for grade, thresholds in self.CLASS_THRESHOLDS.items():
            if total_score >= thresholds['score_min'] and defect_percent <= thresholds['defect_max']:
                return grade
        
        return 'E'  # Default to lowest grade
    
    def _get_dominant_color(self, image: np.ndarray) -> Tuple[int, int, int]:
        """Extract dominant color from image"""
        # Reshape to list of pixels
        pixels = image.reshape(-1, 3)
        
        # Calculate mean color
        mean_color = np.mean(pixels, axis=0)
        
        return tuple(map(int, mean_color))
    
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to hex color"""
        return '#{:02x}{:02x}{:02x}'.format(rgb[2], rgb[1], rgb[0])  # BGR to RGB
    
    def _get_quality_stars(self, score: float) -> str:
        """Get star rating based on score"""
        if score >= 90:
            return '⭐⭐⭐⭐⭐'
        elif score >= 75:
            return '⭐⭐⭐⭐'
        elif score >= 50:
            return '⭐⭐⭐'
        elif score >= 25:
            return '⭐⭐'
        else:
            return '⭐'
    
    def create_analysis_visualization(self, image: np.ndarray, analysis: Dict) -> np.ndarray:
        """Create annotated image with analysis overlay"""
        annotated = image.copy()
        h, w = annotated.shape[:2]
        
        # Create semi-transparent overlay
        overlay = annotated.copy()
        
        # Draw info panel (top-right)
        panel_w = 300
        panel_h = 200
        cv2.rectangle(overlay, (w-panel_w-10, 10), (w-10, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        
        # Title
        cv2.putText(annotated, "BUYO LEAF ANALYSIS", (w-panel_w+10, y_offset),
                   font, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        # Quality grade
        cv2.putText(annotated, f"Grade: CLASS {analysis['quality_grade']}", 
                   (w-panel_w+10, y_offset), font, 0.7, (0, 255, 0), 2)
        y_offset += 25
        
        # Overall score
        cv2.putText(annotated, f"Score: {analysis['overall_score']:.1f}/100", 
                   (w-panel_w+10, y_offset), font, 0.6, (255, 255, 255), 1)
        y_offset += 20
        
        # Stars
        cv2.putText(annotated, analysis['quality_stars'], 
                   (w-panel_w+10, y_offset), font, 0.6, (255, 200, 0), 1)
        y_offset += 25
        
        # Defect info
        defect_color = (0, 255, 0) if analysis['defects']['total_defect_percent'] < 5 else (0, 165, 255)
        if analysis['defects']['total_defect_percent'] > 15:
            defect_color = (0, 0, 255)
        
        cv2.putText(annotated, f"Defects: {analysis['defects']['total_defect_percent']:.1f}%", 
                   (w-panel_w+10, y_offset), font, 0.5, defect_color, 1)
        y_offset += 20
        
        # Color health
        cv2.putText(annotated, f"Color: {analysis['color']['green_ratio']:.1f}% green", 
                   (w-panel_w+10, y_offset), font, 0.4, (255, 255, 255), 1)
        
        return annotated


# Singleton instance
_analyzer_instance = None

def get_analyzer() -> BuyoLeafAnalyzer:
    """Get singleton analyzer instance"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = BuyoLeafAnalyzer()
    return _analyzer_instance

#!/usr/bin/env python3
"""
ID CARD SCANNER - Interactive Improvements (updated)

Changes in this version:
- Main menu compact: values are printed beside entries and aligned.
- "Brief" text moved into submenus and renamed "Description".
- Default mixed pattern set to 'top'.
- PDF copies editable from Output Options.
- Image format option (--image-format) added (png or jpg), editable in Output Options and used for non-PDF outputs.
- Output mode renamed to output-color (CLI and menu).
- Invert alignment wording clarified in Transform submenu.
- Descriptions inside submenus, not on main menu.
- Card-only export always saved as PNG to preserve transparency.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import sys
import os
import math
import argparse
import subprocess
import shutil
import io
import contextlib
import time
import textwrap

# Try to enable readline for improved interactive editing (cursor movement, history)
try:
    import readline  # noqa: F401
except Exception:
    pass

# For tkinter fallback file selection
try:
    import tkinter as tk
    from tkinter import filedialog
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

# Cross-platform single-key read (for TTY menus)
def read_single_key():
    """Read a single key from stdin without waiting for Enter (TTY only). Returns a string (single char) or ''. """
    if not sys.stdin or not sys.stdin.isatty():
        return ''
    try:
        # Windows
        if os.name == 'nt':
            import msvcrt
            ch = msvcrt.getwch()
            return ch
        # Unix
        import tty
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        # Fallback to input (requires Enter)
        try:
            s = input()
            return s[0] if s else ''
        except Exception:
            return ''

# Paper sizes in mm (width x height) - portrait orientation
PAPER_SIZES = {
    'a4': (210, 297),
    'a3': (297, 420),
    'a5': (148, 210),
    'letter': (215.9, 279.4),
    'legal': (215.9, 355.6),
    'tabloid': (279.4, 431.8),
}

# Defaults
OPEN_OUTPUT_FOLDER_DEFAULT = True
GUI_FILE_SELECT_DEFAULT = True
CORNER_AREA_PCT_DEFAULT = 0.20  # Circle area as fraction of image area (20%)
CONTRAST_DEFAULT = 100.0  # percent
BRIGHTNESS_DEFAULT = 100.0  # percent
ORIENTATION_DEFAULT = 'portrait'
IMAGE_FORMAT_DEFAULT = 'png'
MIXED_PATTERN_DEFAULT = 'top'


def get_paper_dimensions(paper_size, orientation='portrait'):
    width, height = PAPER_SIZES[paper_size]
    if orientation == 'landscape':
        return (height, width)
    return (width, height)


def ask_file_native(prompt="Select file"):
    """
    Try platform-native file chooser, fallback to tkinter filedialog if available.
    Returns selected path or None.
    """
    # macOS
    try:
        if sys.platform == "darwin":
            cmd = ['osascript', '-e',
                   'set fn to POSIX path of (choose file with prompt "{}")'.format(prompt.replace('"', '\\"'))]
            out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
    except Exception:
        pass

    # Linux - zenity or kdialog
    try:
        if sys.platform.startswith("linux"):
            if shutil.which("zenity"):
                cmd = ['zenity', '--file-selection', '--title', prompt]
                out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if out.returncode == 0 and out.stdout.strip():
                    return out.stdout.strip()
            if shutil.which("kdialog"):
                cmd = ['kdialog', '--getopenfilename', os.path.expanduser('~'), prompt]
                out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if out.returncode == 0 and out.stdout.strip():
                    return out.stdout.strip()
    except Exception:
        pass

    # Windows - try PowerShell selection
    try:
        if os.name == 'nt':
            ps_script = r'''
Add-Type -AssemblyName System.Windows.Forms
$ofd = New-Object System.Windows.Forms.OpenFileDialog
$ofd.Title = "{}"
$ofd.Filter = "All files (*.*)|*.*"
$null = $ofd.ShowDialog()
if ($ofd.FileName) {{ Write-Output $ofd.FileName }}
'''.format(prompt.replace('"', '\\"'))
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps_script],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
    except Exception:
        pass

    # Fallback to tkinter if available
    if TK_AVAILABLE:
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            file_path = filedialog.askopenfilename(title=prompt)
            root.destroy()
            if file_path:
                return file_path
        except Exception:
            pass

    # If all fail, return None
    return None


def rotate_image(img, degrees):
    """Rotate image by 0/90/180/270 degrees clockwise."""
    if degrees == 0:
        return img
    if degrees == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, degrees, 1.0)
    return cv2.warpAffine(img, M, (w, h))


# --- CornerSelector class with keyboard pixel movement and improved magnifier ---
class CornerSelector:
    def __init__(self, image, title="Select Corners", corner_area_pct=CORNER_AREA_PCT_DEFAULT):
        self.original_image = image.copy()
        self.image = image.copy()
        self.title = title
        self.corners = []
        self.dragging_point = None
        self.last_selected_index = 0  # track last selected corner for keyboard movement
        self.window_name = title
        self.panning = False
        self.pan_start = None
        self.offset_x = 0
        self.offset_y = 0
        self.zoom_scale = 1.0

        # Keep original resolution for better quality
        self.display_image = image.copy()

        # corner selection area percentage
        self.corner_area_pct = corner_area_pct

        # used to keep relative offset when dragging so the grip doesn't jump
        self.drag_offset = [0.0, 0.0]

        # magnifier timing: show 3 seconds after last arrow key press
        self.magnifier_visible_until = 0.0

        # Calculate window size to fit screen
        max_height = 800
        max_width = 1200
        h, w = image.shape[:2]

        if h > max_height or w > max_width:
            scale_h = max_height / h if h > max_height else 1.0
            scale_w = max_width / w if w > max_width else 1.0
            self.window_scale = min(scale_h, scale_w)
        else:
            self.window_scale = 1.0

        # Initialize corners at image corners
        h, w = self.display_image.shape[:2]
        margin = 50
        margin = min(margin, w // 10, h // 10)
        self.corners = [
            [margin, margin],  # Top-left
            [w - margin, margin],  # Top-right
            [w - margin, h - margin],  # Bottom-right
            [margin, h - margin]  # Bottom-left
        ]

        # Auto-zoom calculation
        self.calculate_auto_zoom()

        # Performance heuristics
        self.large_image_threshold = 4000 * 3000  # 12MP threshold
        self.is_large = (self.display_image.shape[0] * self.display_image.shape[1]) > self.large_image_threshold

    def calculate_auto_zoom(self):
        """Calculate zoom to ensure all corners are visible"""
        if not self.corners:
            return
        corners_array = np.array(self.corners)
        min_x, min_y = corners_array.min(axis=0)
        max_x, max_y = corners_array.max(axis=0)
        corner_area = (max_x - min_x) * (max_y - min_y)
        image_area = self.display_image.shape[1] * self.display_image.shape[0]
        if corner_area < image_area * 0.5:
            width_ratio = self.display_image.shape[1] / max(1, (max_x - min_x))
            height_ratio = self.display_image.shape[0] / max(1, (max_y - min_y))
            self.zoom_scale = min(width_ratio, height_ratio) * 0.8
            self.zoom_scale = max(1.0, min(self.zoom_scale, 5.0))
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            self.offset_x = center_x - (self.display_image.shape[1] / (2 * self.zoom_scale))
            self.offset_y = center_y - (self.display_image.shape[0] / (2 * self.zoom_scale))
        else:
            self.zoom_scale = 1.0
            self.offset_x = 0
            self.offset_y = 0

    def transform_point_to_zoom(self, point):
        """Transform point from zoomed coordinates to original image coordinates"""
        x = (point[0] / self.zoom_scale) + self.offset_x
        y = (point[1] / self.zoom_scale) + self.offset_y
        return [x, y]

    def transform_point_from_zoom(self, point):
        """Transform point from original image coordinates to zoomed coordinates"""
        x = (point[0] - self.offset_x) * self.zoom_scale
        y = (point[1] - self.offset_y) * self.zoom_scale
        return [x, y]

    def select_all_image(self):
        """Set corners to the full image extents"""
        h, w = self.display_image.shape[:2]
        self.corners = [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ]
        self.zoom_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

    def _create_soft_circle_mask(self, h, w, radius, soften=10):
        """Create a soft alpha mask for a circle with blurred edges for nicer shadow."""
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (w // 2, h // 2), radius, 255, -1, cv2.LINE_AA)
        k = max(3, int(soften // 2) * 2 + 1)
        blurred = cv2.GaussianBlur(mask, (k, k), soften)
        return blurred.astype(np.float32) / 255.0

    def draw_overlay(self):
        # Create base image with zoom and pan (adaptive interpolation)
        h, w = self.display_image.shape[:2]

        if self.zoom_scale != 1.0:
            zoom_w = int(w / self.zoom_scale)
            zoom_h = int(h / self.zoom_scale)

            start_x = max(0, int(self.offset_x))
            start_y = max(0, int(self.offset_y))
            end_x = min(w, start_x + zoom_w)
            end_y = min(h, start_y + zoom_h)

            if end_x > start_x and end_y > start_y:
                zoom_region = self.display_image[start_y:end_y, start_x:end_x]
                interp = cv2.INTER_LANCZOS4
                if self.is_large:
                    interp = cv2.INTER_LINEAR
                try:
                    zoomed = cv2.resize(zoom_region, (w, h), interpolation=interp)
                except Exception:
                    zoomed = self.display_image.copy()
            else:
                zoomed = self.display_image.copy()
        else:
            zoomed = self.display_image.copy()

        overlay = zoomed.copy()

        # Dark overlay and dimming
        dark_overlay = np.zeros_like(overlay)
        alpha = 0.7
        dimmed = cv2.addWeighted(overlay, 1 - alpha, dark_overlay, alpha, 0)

        # Transform corners to zoomed coordinates for display
        display_corners = [self.transform_point_from_zoom(corner) for corner in self.corners]

        # Create mask for polygon and keep it bright while dimming outside
        mask = np.zeros(overlay.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(display_corners, dtype=np.int32)], 255)
        result = overlay.copy()
        result[mask == 0] = dimmed[mask == 0]

        # Draw polygon border
        pts = np.array(display_corners, dtype=np.int32)
        cv2.polylines(result, [pts], True, (128, 128, 128), 1, cv2.LINE_AA)

        # Visualize corner-selection radius ring
        try:
            h_img, w_img = self.display_image.shape[:2]
            image_area = w_img * h_img
            desired_area = max(0.0, min(1.0, self.corner_area_pct)) * image_area
            radius_px = int(math.sqrt(desired_area / math.pi))
            for corner in display_corners:
                cx, cy = int(corner[0]), int(corner[1])
                cv2.circle(result, (cx, cy), int(radius_px * self.zoom_scale), (200, 200, 200), 1, cv2.LINE_AA)
        except Exception:
            pass

        # Draw corner grips with labels
        for i, corner in enumerate(display_corners):
            color = (0, 0, 255) if self.dragging_point == i else (255, 0, 0)
            cv2.circle(result, tuple(np.int32(corner)), 10, color, -1)
            cv2.circle(result, tuple(np.int32(corner)), 10, (255, 255, 255), 2)
            labels = ['TL', 'TR', 'BR', 'BL']
            cv2.putText(result, labels[i], (int(corner[0] - 20), int(corner[1] - 15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Determine whether to show magnifier:
        show_magnifier = False
        now = time.time()
        if self.dragging_point is not None:
            show_magnifier = True
        elif self.magnifier_visible_until > now:
            show_magnifier = True

        if show_magnifier:
            # pick corner to magnify: prefer dragging_point else last_selected_index
            idx = self.dragging_point if self.dragging_point is not None else self.last_selected_index
            idx = max(0, min(len(self.corners) - 1, idx))
            corner = display_corners[idx]
            zoom_radius = 150
            zoom_factor = 4
            interp = cv2.INTER_LANCZOS4

            cx, cy = int(corner[0]), int(corner[1])

            orig_corner = self.corners[idx]
            orig_cx, orig_cy = int(orig_corner[0]), int(orig_corner[1])

            region_size = zoom_radius // max(1, zoom_factor)
            x1 = max(0, orig_cx - region_size)
            y1 = max(0, orig_cy - region_size)
            x2 = min(self.display_image.shape[1], orig_cx + region_size)
            y2 = min(self.display_image.shape[0], orig_cy + region_size)

            roi = self.display_image[y1:y2, x1:x2].copy()

            if roi.size > 0:
                roi_dimmed = cv2.addWeighted(roi, 1 - alpha, np.zeros_like(roi), alpha, 0)
                test_mask = np.zeros(self.display_image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(test_mask, [np.array(self.corners, dtype=np.int32)], 255)
                roi_mask = test_mask[y1:y2, x1:x2]
                roi_final = roi.copy()
                roi_final[roi_mask == 0] = roi_dimmed[roi_mask == 0]

                zoomed_roi = cv2.resize(roi_final, None, fx=zoom_factor, fy=zoom_factor, interpolation=interp)

                zh, zw = zoomed_roi.shape[:2]
                crop_x1 = max(0, (zw // 2) - zoom_radius)
                crop_y1 = max(0, (zh // 2) - zoom_radius)
                crop_x2 = min(zw, (zw // 2) + zoom_radius)
                crop_y2 = min(zh, (zh // 2) + zoom_radius)

                zoomed_crop = zoomed_roi[crop_y1:crop_y2, crop_x1:crop_x2]

                paste_x1 = max(0, cx - zoom_radius)
                paste_y1 = max(0, cy - zoom_radius)
                paste_x2 = min(result.shape[1], cx + zoom_radius)
                paste_y2 = min(result.shape[0], cy + zoom_radius)

                actual_w = paste_x2 - paste_x1
                actual_h = paste_y2 - paste_y1

                if actual_w > 0 and actual_h > 0:
                    try:
                        zoomed_crop = cv2.resize(zoomed_crop, (actual_w, actual_h), interpolation=interp)
                    except Exception:
                        pass

                    circle_mask = np.zeros((actual_h, actual_w), dtype=np.uint8)
                    mask_cx = actual_w // 2
                    mask_cy = actual_h // 2
                    cv2.circle(circle_mask, (mask_cx, mask_cy),
                              min(zoom_radius, actual_w // 2, actual_h // 2), 255, -1, cv2.LINE_AA)

                    soft_mask = self._create_soft_circle_mask(actual_h, actual_w, min(zoom_radius, actual_w // 2, actual_h // 2), soften=12)
                    shadow = np.zeros_like(zoomed_crop, dtype=np.uint8)
                    shadow[:] = (0, 0, 0)
                    shadow_alpha = (soft_mask * 0.4)[:, :, None]
                    target_area = result[paste_y1:paste_y2, paste_x1:paste_x2]
                    if target_area.shape[:2] == soft_mask.shape:
                        target_area = (target_area * (1 - shadow_alpha) + shadow * shadow_alpha).astype(np.uint8)
                        result[paste_y1:paste_y2, paste_x1:paste_x2] = target_area

                    circle_mask_float = (circle_mask.astype(float) / 255.0)
                    circle_mask_float = circle_mask_float[:, :, None]
                    for c in range(3):
                        result[paste_y1:paste_y2, paste_x1:paste_x2, c] = (
                            circle_mask_float[:, :, 0] * zoomed_crop[:, :, c] +
                            (1 - circle_mask_float[:, :, 0]) * result[paste_y1:paste_y2, paste_x1:paste_x2, c]
                        ).astype(np.uint8)

                    cv2.circle(result, (cx, cy),
                              min(zoom_radius, actual_w // 2, actual_h // 2),
                              (255, 255, 255), 3, cv2.LINE_AA)
                    cv2.circle(result, (cx, cy),
                              min(zoom_radius, actual_w // 2, actual_h // 2),
                              (0, 0, 0), 1, cv2.LINE_AA)

                    crosshair_size = 20
                    cv2.line(result, (cx - crosshair_size, cy), (cx + crosshair_size, cy),
                            (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.line(result, (cx, cy - crosshair_size), (cx, cy + crosshair_size),
                            (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.circle(result, (cx, cy), 4, (0, 255, 0), -1, cv2.LINE_AA)

        if self.zoom_scale != 1.0:
            zoom_text = f"Zoom: {self.zoom_scale:.1f}x"
            cv2.putText(result, zoom_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result, "Middle drag: Pan | Scroll: Zoom", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if self.window_scale != 1.0:
            display_h = int(result.shape[0] * self.window_scale)
            display_w = int(result.shape[1] * self.window_scale)
            result = cv2.resize(result, (display_w, display_h), interpolation=cv2.INTER_AREA)

        return result

    def mouse_callback(self, event, x, y, flags, param):
        x = int(x / self.window_scale)
        y = int(y / self.window_scale)
        orig_x, orig_y = self.transform_point_to_zoom([x, y])

        if event == cv2.EVENT_LBUTTONDOWN:
            h, w = self.display_image.shape[:2]
            image_area = w * h
            desired_area = max(0.0, min(1.0, self.corner_area_pct)) * image_area
            radius_px = int(math.sqrt(desired_area / math.pi))
            threshold = max(10, radius_px) / max(1.0, self.zoom_scale)

            nearest_i = None
            nearest_dist = float('inf')
            for i, corner in enumerate(self.corners):
                dist = math.hypot(orig_x - corner[0], orig_y - corner[1])
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_i = i

            if nearest_dist <= threshold:
                self.dragging_point = nearest_i
                self.last_selected_index = nearest_i
                corner = self.corners[self.dragging_point]
                self.drag_offset = [corner[0] - orig_x, corner[1] - orig_y]
                self.panning = False

        elif event == cv2.EVENT_MBUTTONDOWN:
            self.panning = True
            self.pan_start = [x, y]

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_point is not None:
                h, w = self.display_image.shape[:2]
                new_x = orig_x + self.drag_offset[0]
                new_y = orig_y + self.drag_offset[1]
                constrained_x = max(0, min(new_x, w - 1))
                constrained_y = max(0, min(new_y, h - 1))
                self.corners[self.dragging_point] = [constrained_x, constrained_y]
            elif self.panning and self.pan_start is not None:
                dx = (self.pan_start[0] - x) / self.zoom_scale
                dy = (self.pan_start[1] - y) / self.zoom_scale

                self.offset_x += dx
                self.offset_y += dy

                h, w = self.display_image.shape[:2]
                max_offset_x = max(0, w - (w / self.zoom_scale))
                max_offset_y = max(0, h - (h / self.zoom_scale))

                self.offset_x = max(0, min(self.offset_x, max_offset_x))
                self.offset_y = max(0, min(self.offset_y, max_offset_y))

                self.pan_start = [x, y]

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_point = None
            self.drag_offset = [0.0, 0.0]

        elif event == cv2.EVENT_MBUTTONUP:
            self.panning = False
            self.pan_start = None

        elif event == cv2.EVENT_MOUSEWHEEL:
            zoom_delta = 0.1 if flags > 0 else -0.1
            old_zoom = self.zoom_scale
            self.zoom_scale = max(1.0, min(10.0, self.zoom_scale + zoom_delta))

            if self.zoom_scale != old_zoom:
                zoom_ratio = self.zoom_scale / old_zoom
                self.offset_x = orig_x - (orig_x - self.offset_x) / zoom_ratio
                self.offset_y = orig_y - (orig_y - self.offset_y) / zoom_ratio

                h, w = self.display_image.shape[:2]
                max_offset_x = max(0, w - (w / self.zoom_scale))
                max_offset_y = max(0, h - (h / self.zoom_scale))
                self.offset_x = max(0, min(self.offset_x, max_offset_x))
                self.offset_y = max(0, min(self.offset_y, max_offset_y))

    def select(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print(f"\n{self.title}:")
        print("- Drag the corner points to select card boundaries")
        print("- Middle mouse button drag to pan when zoomed")
        print("- Mouse wheel to zoom in/out")
        print("- Press ENTER when done")
        print("- Press R to reset corners")
        print("- Press Z to auto-zoom to corners")
        print("- Press A to select the entire image")
        print("- Press ESC to cancel")
        print("- Use arrow keys to move the last selected corner pixel-by-pixel; magnifier shows for 3 seconds after last arrow press")

        # Key codes to handle arrow keys across platforms
        LEFT_KEYS = (81, 2424832)
        UP_KEYS = (82, 2490368)
        RIGHT_KEYS = (83, 2555904)
        DOWN_KEYS = (84, 2621440)

        while True:
            display = self.draw_overlay()
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1)
            # handle Enter (13) and ESC (27)
            if key == -1:
                continue

            if (key & 0xFF) == 13:
                break
            if (key & 0xFF) == 27:
                cv2.destroyWindow(self.window_name)
                return None
            # reset
            if (key & 0xFF) in (ord('r'), ord('R')):
                h, w = self.display_image.shape[:2]
                margin = 50
                margin = min(margin, w // 10, h // 10)
                self.corners = [
                    [margin, margin],
                    [w - margin, margin],
                    [w - margin, h - margin],
                    [margin, h - margin]
                ]
                self.zoom_scale = 1.0
                self.offset_x = 0
                self.offset_y = 0
            # auto-zoom
            elif (key & 0xFF) in (ord('z'), ord('Z')):
                self.calculate_auto_zoom()
            # select all
            elif (key & 0xFF) in (ord('a'), ord('A')):
                self.select_all_image()
            else:
                # Arrow keys for pixel movement - support multiple codes
                if key in LEFT_KEYS or key in UP_KEYS or key in RIGHT_KEYS or key in DOWN_KEYS or (key & 0xFF) in (81,82,83,84):
                    low = key & 0xFF
                    if low in (81, 82, 83, 84):
                        kcode = low
                    else:
                        kcode = key
                    idx = self.dragging_point if self.dragging_point is not None else self.last_selected_index
                    idx = max(0, min(len(self.corners) - 1, idx))
                    dx = 0
                    dy = 0
                    if kcode in LEFT_KEYS or kcode == 81:
                        dx = -1
                    elif kcode in RIGHT_KEYS or kcode == 83:
                        dx = 1
                    elif kcode in UP_KEYS or kcode == 82:
                        dy = -1
                    elif kcode in DOWN_KEYS or kcode == 84:
                        dy = 1
                    if dx != 0 or dy != 0:
                        h_img, w_img = self.display_image.shape[:2]
                        new_x = max(0, min(w_img - 1, int(self.corners[idx][0] + dx)))
                        new_y = max(0, min(h_img - 1, int(self.corners[idx][1] + dy)))
                        self.corners[idx] = [new_x, new_y]
                        self.last_selected_index = idx
                        self.magnifier_visible_until = time.time() + 3.0
                        self.calculate_auto_zoom()

        cv2.destroyWindow(self.window_name)
        return np.array(self.corners, dtype=np.float32)


# --- End of CornerSelector ---

def perspective_transform(image, corners, output_width, output_height):
    """Apply perspective transform to extract card"""
    dst_corners = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(corners, dst_corners)
    warped = cv2.warpPerspective(image, matrix, (output_width, output_height),
                                 flags=cv2.INTER_LANCZOS4)
    return warped


def resize_to_card_dimensions(image, output_width, output_height):
    """Resize image to card dimensions"""
    return cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)


def add_rounded_corners(image, radius_mm=3.18, dpi=300, transparent=False):
    """Add rounded corners to image (3.18mm is standard for credit cards)"""
    radius_px = int((radius_mm / 25.4) * dpi)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mask = Image.new('L', pil_image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), pil_image.size], radius=radius_px, fill=255)

    if transparent:
        output = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        pil_image = pil_image.convert('RGBA')
        output.paste(pil_image, (0, 0), mask)
        return output
    else:
        output = Image.new('RGB', pil_image.size, 'white')
        output.paste(pil_image, (0, 0), mask)
        return cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)


def calculate_max_grid(card_width_px, card_height_px, page_width_px, page_height_px):
    """Calculate maximum grid size that fits on page"""
    max_cols = int(page_width_px / card_width_px)
    max_rows = int(page_height_px / card_height_px)
    return max(1, max_rows), max(1, max_cols)


def create_a4_layout(front_card, back_card, page_width_px, page_height_px):
    """Create page with both cards in vertical layout"""
    page = np.ones((page_height_px, page_width_px, 3), dtype=np.uint8) * 255
    card_height, card_width = front_card.shape[:2]
    x_offset = (page_width_px - card_width) // 2
    spacing = (page_height_px - 2 * card_height) // 3
    y_front = spacing
    y_back = spacing * 2 + card_height
    if y_front >= 0 and y_front + card_height <= page_height_px:
        page[y_front:y_front + card_height, x_offset:x_offset + card_width] = front_card
    if y_back >= 0 and y_back + card_height <= page_height_px:
        page[y_back:y_back + card_height, x_offset:x_offset + card_width] = back_card
    return page


def create_array_layout(cards, rows, cols, page_width_px, page_height_px,
                        align_h=None, align_v=None, margin_px=0):
    """Create page with array of cards distributed with optional alignment."""
    page = np.ones((page_height_px, page_width_px, 3), dtype=np.uint8) * 255
    if not cards:
        return page

    card_height, card_width = cards[0].shape[:2]
    total_width = cols * card_width
    total_height = rows * card_height

    avail_w = page_width_px - total_width
    if avail_w < 0:
        print("Warning: cards don't fit horizontally on page.")
        return page

    avail_h = page_height_px - total_height
    if avail_h < 0:
        print("Warning: cards don't fit vertically on page.")
        return page

    if align_h == 'left':
        if margin_px * 2 <= page_width_px - total_width:
            start_x = margin_px
            remaining_w = page_width_px - total_width - 2 * margin_px
            spacing_x = int(remaining_w / (cols - 1)) if cols > 1 else 0
        else:
            start_x = int(avail_w / 2)
            spacing_x = int(avail_w / (cols + 1))
    elif align_h == 'right':
        if margin_px * 2 <= page_width_px - total_width:
            start_x = page_width_px - margin_px - total_width
            remaining_w = page_width_px - total_width - 2 * margin_px
            spacing_x = int(remaining_w / (cols - 1)) if cols > 1 else 0
        else:
            start_x = int(avail_w / 2)
            spacing_x = int(avail_w / (cols + 1))
    else:
        spacing_x = int(avail_w / (cols + 1)) if cols > 0 else 0
        start_x = int(spacing_x)

    if align_v == 'top':
        if margin_px * 2 <= page_height_px - total_height:
            start_y = margin_px
            remaining_h = page_height_px - total_height - 2 * margin_px
            spacing_y = int(remaining_h / (rows - 1)) if rows > 1 else 0
        else:
            start_y = int(avail_h / 2)
            spacing_y = int(avail_h / (rows + 1))
    elif align_v == 'bottom':
        if margin_px * 2 <= page_height_px - total_height:
            start_y = page_height_px - margin_px - total_height
            remaining_h = page_height_px - total_height - 2 * margin_px
            spacing_y = int(remaining_h / (rows - 1)) if rows > 1 else 0
        else:
            start_y = int(avail_h / 2)
            spacing_y = int(avail_h / (rows + 1))
    else:
        spacing_y = int(avail_h / (rows + 1)) if rows > 0 else 0
        start_y = int(spacing_y)

    card_idx = 0
    for r in range(rows):
        for c in range(cols):
            if card_idx >= len(cards):
                break
            x_pos = start_x + c * (card_width + spacing_x)
            y_pos = start_y + r * (card_height + spacing_y)
            if (y_pos + card_height <= page_height_px and
                    x_pos + card_width <= page_width_px and
                    y_pos >= 0 and x_pos >= 0):
                page[y_pos:y_pos + card_height, x_pos:x_pos + card_width] = cards[card_idx]
            card_idx += 1
        if card_idx >= len(cards):
            break

    return page


def create_mixed_array_layout(front_card, back_card, rows, cols, page_width_px, page_height_px, pattern='checker',
                              align_h=None, align_v=None, margin_px=0):
    total_cards = rows * cols
    cards = []
    for r in range(rows):
        for c in range(cols):
            use_front = False
            if pattern == 'checker':
                use_front = ((r + c) % 2 == 0)
            elif pattern == 'left':
                use_front = (c % 2 == 0)
            elif pattern == 'right':
                use_front = (c % 2 == 1)
            elif pattern == 'top':
                use_front = (r % 2 == 0)
            elif pattern == 'bottom':
                use_front = (r % 2 == 1)
            cards.append(front_card if use_front else back_card)
    return create_array_layout(cards, rows, cols, page_width_px, page_height_px, align_h=align_h, align_v=align_v,
                               margin_px=margin_px)


def save_as_pdf(images, output_path, dpi=300):
    """Save images as PDF with multiple pages."""
    try:
        from PIL import Image as PILImage

        pil_images = []
        for img in images:
            if isinstance(img, Image.Image):
                pil_img = img.convert("RGB")
            else:
                arr = img
                if arr.dtype != np.uint8:
                    arr = (np.clip(arr, 0, 255)).astype(np.uint8)

                if arr.ndim == 2:
                    pil_img = PILImage.fromarray(arr).convert("RGB")
                elif arr.shape[2] == 3:
                    pil_img = PILImage.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)).convert("RGB")
                elif arr.shape[2] == 4:
                    rgba = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
                    temp = PILImage.fromarray(rgba)
                    bg = PILImage.new("RGB", temp.size, (255, 255, 255))
                    bg.paste(temp, mask=temp.split()[3])
                    pil_img = bg
                else:
                    pil_img = PILImage.fromarray(arr).convert("RGB")

            pil_images.append(pil_img)

        if len(pil_images) > 0:
            devnull = open(os.devnull, "w")
            try:
                with contextlib.redirect_stderr(devnull):
                    pil_images[0].save(
                        output_path,
                        save_all=True,
                        append_images=pil_images[1:] if len(pil_images) > 1 else [],
                        resolution=dpi,
                        quality=95
                    )
            finally:
                devnull.close()
            return True
    except Exception as e:
        print(f"Error saving PDF: {e}")
        return False
    return False


def open_path(path):
    """Open a file or folder in OS default application / file manager"""
    try:
        if os.path.isdir(path):
            if os.name == 'nt':
                os.startfile(path)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', path])
            else:
                subprocess.Popen(['xdg-open', path])
        else:
            if os.name == 'nt':
                os.startfile(path)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', path])
            else:
                subprocess.Popen(['xdg-open', path])
    except Exception as e:
        print(f"Could not open {path}: {e}")


def apply_brightness_contrast(img, brightness_percent, contrast_percent):
    """Apply brightness and contrast where values are percentages (100 = unchanged)."""
    try:
        b = float(brightness_percent) / 100.0
        c = float(contrast_percent) / 100.0
        arr = img.astype(np.float32)
        arr = (arr - 128.0) * c + 128.0
        arr = arr * b
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    except Exception:
        return img


def apply_output_color(img, color_mode, contrast_percent=CONTRAST_DEFAULT, brightness_percent=BRIGHTNESS_DEFAULT):
    """Apply brightness/contrast then output color transformation (expects BGR numpy array)"""
    img = apply_brightness_contrast(img, brightness_percent, contrast_percent)

    if color_mode == 'color':
        return img
    elif color_mode == 'grayscale':
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    elif color_mode == 'blackwhite':
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    elif color_mode == 'enhanced':
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return enhanced
    else:
        return img


def ask_file_gui(prompt="Select file"):
    """Wrapper that prefers native dialogs but falls back to tkinter."""
    path = ask_file_native(prompt)
    if path:
        return path
    if TK_AVAILABLE:
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            file_path = filedialog.askopenfilename(title=prompt)
            root.destroy()
            if file_path:
                return file_path
        except Exception:
            pass
    return None


def build_arg_parser(script_path):
    p = argparse.ArgumentParser(
        description="ID Card Scanner: extract and layout ID/credit-card sized images.",
        epilog=f"Tips: Run the script using the full path if launching from another directory, e.g. '{script_path} front.jpg back.jpg -o /full/path/outname'."
    )
    p.add_argument('front', nargs='?', help='Front side image path')
    p.add_argument('back', nargs='?', help='Back side image path')
    p.add_argument('-o', '--output', help='Output file base path (no extension)', default='id_card_output')
    p.add_argument('--dpi', type=int, default=300, help='DPI for output (default 300)')
    p.add_argument('--output-color', choices=['color', 'grayscale', 'enhanced', 'blackwhite'],
                   default='color', help='Output color mode (default color)')
    p.add_argument('--contrast-percent', type=float, default=CONTRAST_DEFAULT,
                   help='Contrast percentage (100 = unchanged).')
    p.add_argument('--brightness-percent', type=float, default=BRIGHTNESS_DEFAULT,
                   help='Brightness percentage (100 = unchanged).')
    p.add_argument('--no-crop', action='store_true', help='Skip cropping, use entire image (stretch to fit)')
    p.add_argument('--card-only', action='store_true', help='Export card only with transparent background (PNG)')
    p.add_argument('--array', choices=['separate', 'mixed'], help='Create array layout')
    p.add_argument('--grid', nargs='+', type=str, metavar=('ROWS', 'COLS'),
                   help="Grid dimensions (rows cols). Provide one token to apply to both dimensions (e.g. '3' or 'max'), "
                        "or two tokens 'ROWS COLS'. Each token can be an integer or 'max'. Examples: --grid max 3, --grid 2, --grid max")
    p.add_argument('--paper', choices=list(PAPER_SIZES.keys()), default='a4', help='Paper size (default a4)')
    p.add_argument('--orientation', choices=['portrait', 'landscape'], default=ORIENTATION_DEFAULT,
                   help='Page orientation (portrait/landscape)')
    p.add_argument('--pdf', nargs='?', const=1, type=int, help='Output as PDF with optional number of copies')
    p.add_argument('--pdf-copies', nargs='?', const=1, type=int, help='Shorthand to set number of PDF copies (overrides --pdf if provided)')
    p.add_argument('--no-open-folder', action='store_true', help='Do not open export folder after finish (when multiple files)')
    p.add_argument('--no-open-after', action='store_true', help='Do not open exported file or folder after finish')
    p.add_argument('--no-gui-select', action='store_true', help='Do not use GUI file selector (use CLI paths/input)')
    p.add_argument('--corner-area-pct', type=float, default=CORNER_AREA_PCT_DEFAULT,
                   help='Corner selector circle area as fraction of image area (default 0.20 for 20%%)')
    p.add_argument('--image-format', choices=['png', 'jpg'], default=IMAGE_FORMAT_DEFAULT,
                   help='Output image format for non-PDF exports (png/jpg)')
    p.add_argument('--mixed-pattern', choices=['checker', 'left', 'right', 'top', 'bottom'],
                   default=MIXED_PATTERN_DEFAULT, help="Pattern for mixed array arrangement")
    p.add_argument('--align', nargs='+', type=str, choices=['left', 'right', 'top', 'bottom'],
                   help="Align grid to edges when grid is smaller than page. Provide one or two tokens (e.g. '--align left' or '--align left top').")
    p.add_argument('--align-margin-mm', type=float, default=10.0,
                   help="Margin in millimeters used for alignment from page edge (default 10.0 mm)")
    p.add_argument('--rotate', nargs='*',
                   help="Rotation tokens like 'image=90', 'image-front=90', 'page-back=180'. "
                        "Allowed keys: image, image-front, image-back, page, page-front, page-back. "
                        "Allowed degrees: 0,90,180,270. Per-side keys override global keys.")
    p.add_argument('--invert-align-h', action='store_true',
                   help="Alternate/invert horizontal alignment (left<->right) on every page based on --align horizontal value.")
    p.add_argument('--invert-align-v', action='store_true',
                   help="Alternate/invert vertical alignment (top<->bottom) on every page based on --align vertical value.")
    p.add_argument('-m', '--menu', action='store_true', help='Force interactive menu even if input files provided')
    return p


def prompt_yes_no(prompt, default_yes=True):
    """Prompt user for Y/n question. Returns True/False. If not interactive, default supplied value."""
    if not sys.stdin or not sys.stdin.isatty():
        return default_yes
    try:
        sys.stdout.write(f"{prompt} [{'Y/n' if default_yes else 'y/N'}]: ")
        sys.stdout.flush()
        ch = read_single_key()
        if ch == '':
            resp = input().strip().lower()
            if resp == '':
                return default_yes
            return resp in ('y', 'yes')
        sys.stdout.write(f"{ch}\n")
        sys.stdout.flush()
        if ch.lower() in ('y', 'n'):
            return ch.lower() == 'y'
        if ch == '\r' or ch == '\n':
            return default_yes
        return ch.lower() in ('y',)
    except Exception:
        return default_yes


def parse_grid_token(tok, name):
    """Parse a single grid token, return None for 'max' or integer value. Raise ValueError if invalid."""
    if tok is None:
        return None
    t = tok.strip().lower()
    if t == 'max':
        return None
    try:
        v = int(t)
        if v <= 0:
            raise ValueError(f"{name} must be positive")
        return v
    except ValueError:
        raise ValueError(f"Invalid grid token for {name}: '{tok}' (must be integer or 'max')")


def validate_align_tokens(tokens):
    """Validate align tokens: accept 1 or 2 tokens, not contradictory. Return (align_h, align_v)."""
    if not tokens:
        return None, None
    if len(tokens) > 2:
        raise ValueError("Provide at most two align tokens (one horizontal and/or one vertical).")
    toks = [t.lower() for t in tokens]
    if 'left' in toks and 'right' in toks:
        raise ValueError("Cannot specify both 'left' and 'right'.")
    if 'top' in toks and 'bottom' in toks:
        raise ValueError("Cannot specify both 'top' and 'bottom'.")
    align_h = None
    align_v = None
    for t in toks:
        if t in ('left', 'right'):
            if align_h is not None:
                raise ValueError("Multiple horizontal align tokens provided.")
            align_h = t
        elif t in ('top', 'bottom'):
            if align_v is not None:
                raise ValueError("Multiple vertical align tokens provided.")
            align_v = t
        else:
            raise ValueError(f"Unknown align token: {t}")
    return align_h, align_v


def flip_align(align):
    if align == 'left':
        return 'right'
    if align == 'right':
        return 'left'
    if align == 'top':
        return 'bottom'
    if align == 'bottom':
        return 'top'
    return None


def parse_rotation_tokens(tokens):
    """Parse rotation tokens of the form key=deg and return dict mapping keys to deg."""
    if not tokens:
        return {}
    allowed_keys = {'image', 'image-front', 'image-back', 'page', 'page-front', 'page-back'}
    allowed_degs = {0, 90, 180, 270}
    result = {}
    for tok in tokens:
        if '=' not in tok:
            print(f"Warning: Ignoring invalid rotate token '{tok}' (expected key=deg).")
            continue
        key, val = tok.split('=', 1)
        key = key.strip().lower()
        val = val.strip()
        if key not in allowed_keys:
            print(f"Warning: Ignoring unknown rotate key '{key}'. Allowed keys: {', '.join(sorted(allowed_keys))}.")
            continue
        try:
            deg = int(val)
        except Exception:
            print(f"Warning: Ignoring rotate token with non-integer degree '{tok}'.")
            continue
        if deg not in allowed_degs:
            print(f"Warning: Ignoring rotate token with invalid degree '{deg}' (allowed: 0,90,180,270).")
            continue
        result[key] = deg
    return result


def parse_align_string(s):
    if not s:
        return None, None
    tok = s.strip().lower()
    if tok in ('center', '', 'none'):
        return None, None
    parts = tok.split()
    h = None
    v = None
    for p in parts:
        if p in ('left', 'right'):
            h = p
        elif p in ('top', 'bottom'):
            v = p
    return h, v


def align_string_from_parts(ah, av):
    parts = []
    if av:
        parts.append(av)
    if ah:
        parts.append(ah)
    if not parts:
        return 'center'
    return ' '.join(parts)


def shlex_quote(s):
    if not s:
        return "''"
    if ' ' in s or '"' in s or "'" in s:
        return '"' + s.replace('"', '\\"') + '"'
    return s


def build_cli_equivalent(state):
    """Build a CLI command equivalent to current state (best-effort)."""
    parts = [shlex_quote(sys.argv[0])]
    if state.get('front'):
        parts.append(shlex_quote(state.get('front')))
    if state.get('back'):
        parts.append(shlex_quote(state.get('back')))
    parts.extend(['-o', shlex_quote(state.get('output_base'))])
    parts.extend(['--dpi', str(state.get('dpi'))])
    parts.extend(['--output-color', state.get('output_color', state.get('output_mode', 'color'))])
    parts.extend(['--contrast-percent', str(state.get('contrast_percent', CONTRAST_DEFAULT))])
    parts.extend(['--brightness-percent', str(state.get('brightness_percent', BRIGHTNESS_DEFAULT))])
    parts.extend(['--image-format', state.get('image_format', IMAGE_FORMAT_DEFAULT)])
    if state.get('no_crop'):
        parts.append('--no-crop')
    if state.get('card_only'):
        parts.append('--card-only')
    if state.get('array'):
        parts.extend(['--array', state.get('array')])
    if state.get('grid_spec'):
        parts.append('--grid')
        parts.extend(state.get('grid_spec'))
    parts.extend(['--paper', state.get('paper')])
    parts.extend(['--orientation', state.get('orientation')])
    if state.get('pdf'):
        parts.extend(['--pdf', str(state.get('pdf'))])
    if state.get('pdf_copies'):
        parts.extend(['--pdf-copies', str(state.get('pdf_copies'))])
    if not state.get('gui_file_select', True):
        parts.append('--no-gui-select')
    if state.get('rotate_map'):
        for k, v in state.get('rotate_map', {}).items():
            parts.extend(['--rotate', f"{k}={v}"])
    if state.get('invert_h_flag'):
        parts.append('--invert-align-h')
    if state.get('invert_v_flag'):
        parts.append('--invert-align-v')
    if state.get('mixed_pattern'):
        parts.extend(['--mixed-pattern', state.get('mixed_pattern')])
    ah, av = parse_align_string(state.get('align')) if state.get('align') else (None, None)
    if ah or av:
        parts.append('--align')
        if ah:
            parts.append(ah)
        if av:
            parts.append(av)
    return ' '.join(parts)


def interactive_config_review(state, parser=None):
    """
    Main menu with grouped submenus and compact single-line entries.
    """
    if not sys.stdin or not sys.stdin.isatty():
        return state

    def compute_max_grid_for_state(st):
        card_width_mm = 85.6
        card_height_mm = 53.98
        dpi_local = st.get('dpi', 300)
        card_w_px = int((card_width_mm / 25.4) * dpi_local)
        card_h_px = int((card_height_mm / 25.4) * dpi_local)
        paper_w_mm, paper_h_mm = get_paper_dimensions(st.get('paper', 'a4'), st.get('orientation', 'portrait'))
        page_w_px = int((paper_w_mm / 25.4) * dpi_local)
        page_h_px = int((paper_h_mm / 25.4) * dpi_local)
        max_r, max_c = calculate_max_grid(card_w_px, card_h_px, page_w_px, page_h_px)
        return max_r, max_c

    def grid_spec_display(st):
        gs = st.get('grid_spec')
        max_r, max_c = compute_max_grid_for_state(st)
        if not gs:
            return f"Auto(max {max_r}x{max_c})"
        try:
            if len(gs) == 1:
                token = gs[0].lower()
                if token == 'max':
                    return f"{max_r}x{max_c}(max)"
                else:
                    v = int(token)
                    return f"{v}x{v}(max {max_r}x{max_c})"
            else:
                r_tok = gs[0].lower()
                c_tok = gs[1].lower()
                r_str = f"{max_r}(max)" if r_tok == 'max' else r_tok
                c_str = f"{max_c}(max)" if c_tok == 'max' else c_tok
                return f"{r_str}x{c_str}(page max {max_r}x{max_c})"
        except Exception:
            return str(gs)

    # File selection flow
    def file_selection_flow(which):
        current = state.get(which)
        print(f"\n[{which.capitalize()} image] Provide a path to the {which.upper()} image.")
        if current:
            print(f"Current: {current}")
            sys.stdout.write("Press Enter to keep, 'c' to change, 'b' to go back: ")
            sys.stdout.flush()
            key = read_single_key()
            if key == '':
                try:
                    txt = input().strip()
                except Exception:
                    txt = ''
                if txt == '':
                    return current
                if txt.lower() == 'c':
                    key = 'c'
                elif txt.lower() == 'b':
                    return None
                else:
                    return txt
            if key in ('\r', '\n'):
                print()
                return current
            if key in ('\x1b',):
                print()
                return None
            if key.lower() == 'b':
                print()
                return None
            if key.lower() == 'c':
                print()
            else:
                print(key, end='', flush=True)
                try:
                    rest = input().strip()
                    candidate = (key + rest).strip()
                except Exception:
                    candidate = key
                if candidate == '':
                    if state.get('gui_file_select', True):
                        sel = ask_file_gui(f"Select {which.upper()} side image")
                        if sel:
                            return sel
                        else:
                            return None
                    else:
                        print("GUI file selection disabled and no path provided.")
                        return None
                return candidate

        print("Enter file path and press Enter.")
        print("Leave empty and press Enter to open GUI file selector (or press 'b' then Enter to go back).")
        v = input(f"Path (current: {current or '(none)'}): ").strip()
        if v.lower() == 'b':
            return None
        if v == '':
            if state.get('gui_file_select', True):
                sel = ask_file_gui(f"Select {which.upper()} side image")
                if sel:
                    return sel
                else:
                    return None
            else:
                print("GUI file selection disabled and no path provided.")
                return None
        return v

    # Transform menu
    def transform_menu():
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("=== Transform Options ===")
            print("1) Orientation             :", state.get('orientation'))
            print("   Description: Page orientation affects page dimensions (portrait/landscape).")
            print("2) Image/Page Rotation    :", state.get('rotate_map') or "(none)")
            print("   Description: Apply preset rotations (0/90/180/270) to images or pages.")
            print("3) Invert Alignment H     :", state.get('invert_h_flag'))
            print("   Description: Invert alignment every page horizontally (left<->right) on alternating pages.")
            print("4) Invert Alignment V     :", state.get('invert_v_flag'))
            print("   Description: Invert alignment every page vertically (top<->bottom) on alternating pages.")
            print("5) Align                  :", state.get('align') or "center")
            print("   Description: Choose final alignment of the grid on the page (e.g. 'top left').")
            print()
            print("Enter empty to return to main menu.")
            sys.stdout.write("Choose (single key): ")
            sys.stdout.flush()
            ch = read_single_key()
            if ch == '':
                try:
                    line = input()
                except Exception:
                    line = ''
                if line.strip() == '':
                    return
                ch = line.strip()[0]
            if ch in ('\r', '\n'):
                return
            k = ch.lower()

            if k == '1':
                print("\n[Orientation] Choose orientation: 1) portrait  2) landscape")
                sel = read_single_key()
                if sel == '':
                    sel = input("Choose (1-2): ").strip()
                else:
                    print(sel)
                if sel == '1':
                    state['orientation'] = 'portrait'
                elif sel == '2':
                    state['orientation'] = 'landscape'
                else:
                    print("Invalid selection.")
                time.sleep(0.2)
                continue

            if k == '2':
                print("\n[Rotation] Set rotation for image or page scopes. Single-key choices auto-accept.")
                print("Scopes: i) image  f) image-front  b) image-back  p) page  1) page-front  2) page-back  x) none")
                s = read_single_key()
                if s == '':
                    s = input("Choose scope key: ").strip()
                else:
                    print(s)
                mapping = {'i': 'image', 'f': 'image-front', 'b': 'image-back', 'p': 'page', '1': 'page-front', '2': 'page-back'}
                if not s or s.lower() == 'x':
                    print("Skipping rotation step.")
                    time.sleep(0.2)
                    continue
                if s.lower() not in mapping:
                    print("Invalid scope.")
                    time.sleep(0.2)
                    continue
                scope_key = mapping[s.lower()]
                print("Choose degree (single key): 1)0  2)90  3)180  4)270")
                d = read_single_key()
                if d == '':
                    d = input("Choose degree key: ").strip()
                else:
                    print(d)
                deg_map = {'1': 0, '2': 90, '3': 180, '4': 270}
                if d in deg_map:
                    rm = state.get('rotate_map', {}).copy()
                    rm[scope_key] = deg_map[d]
                    state['rotate_map'] = rm
                    print(f"Set {scope_key} rotation to {deg_map[d]}°")
                else:
                    print("Invalid degree selection.")
                time.sleep(0.2)
                continue

            if k == '3':
                state['invert_h_flag'] = not state.get('invert_h_flag', False)
                print("Invert Alignment every page Horizontally:", state['invert_h_flag'])
                time.sleep(0.2)
                continue

            if k == '4':
                state['invert_v_flag'] = not state.get('invert_v_flag', False)
                print("Invert Alignment every page Vertically:", state['invert_v_flag'])
                time.sleep(0.2)
                continue

            if k == '5':
                print("\n[Align] Choose alignment (single key; auto-accept).")
                opts = [
                    ('0', 'center'),
                    ('1', 'top left'),
                    ('2', 'top'),
                    ('3', 'top right'),
                    ('4', 'left'),
                    ('5', 'right'),
                    ('6', 'bottom left'),
                    ('7', 'bottom'),
                    ('8', 'bottom right'),
                ]
                for key, txt in opts:
                    print(f" {key}) {txt}   - Description: positions the grid when it doesn't fill the page.")
                sel = read_single_key()
                if sel == '':
                    sel = input("Choose align key: ").strip()
                else:
                    print(sel)
                mapping = {k: v for k, v in opts}
                if sel in mapping:
                    chosen = mapping[sel]
                    state['align'] = None if chosen == 'center' else chosen
                    print("Align set to:", state['align'] or 'center')
                else:
                    print("Invalid selection.")
                time.sleep(0.2)
                continue

            print("Unknown selection in Transform menu.")
            time.sleep(0.2)

    # Output Options menu
    def output_options_menu():
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("=== Output Options ===")
            print("1) DPI                       :", state.get('dpi'))
            print("   Description: DPI used to compute pixel sizes for card and page.")
            print("2) Output Color              :", state.get('output_color', state.get('output_mode', 'color')))
            print("   Description: Choose color processing mode (color/grayscale/enhanced/blackwhite).")
            print("3) Brightness / Contrast     :", f"{state.get('brightness_percent', BRIGHTNESS_DEFAULT)}% / {state.get('contrast_percent', state.get('contrast_level', CONTRAST_DEFAULT))}%")
            print("   Description: Percent values; 100% = unchanged.")
            print("4) Card-only export          :", state.get('card_only'))
            print("   Description: Export just the card as PNG with rounded corners and transparency.")
            print("5) Array Options             :", state.get('array'))
            print("   Description: Configure array mode, grid and mixed pattern (move grid here).")
            print("6) PDF copies                :", state.get('pdf') or state.get('pdf_copies'))
            print("   Description: Number of PDF copies/pages when exporting as PDF.")
            print("7) Image format              :", state.get('image_format', IMAGE_FORMAT_DEFAULT))
            print("   Description: Output format for non-PDF exports (png/jpg). Card-only stays PNG.")
            print()
            print("Enter empty to return to main menu.")
            sys.stdout.write("Choose (single key): ")
            sys.stdout.flush()
            ch = read_single_key()
            if ch == '':
                try:
                    line = input()
                except Exception:
                    line = ''
                if line.strip() == '':
                    return
                ch = line.strip()[0]
            if ch in ('\r', '\n'):
                return
            k = ch.lower()

            if k == '1':
                while True:
                    v = input(f"Enter DPI (current: {state.get('dpi')}): ").strip()
                    if v == '':
                        break
                    try:
                        dpiv = int(v)
                        if dpiv <= 0:
                            raise ValueError()
                        state['dpi'] = dpiv
                        break
                    except Exception:
                        print("Please enter a positive integer for DPI.")
                continue

            if k == '2':
                choices = ['color', 'grayscale', 'enhanced', 'blackwhite']
                for i, c in enumerate(choices, start=1):
                    print(f" {i}) {c}")
                ch2 = read_single_key()
                if ch2 == '':
                    v = input(f"Choose mode number (current: {state.get('output_color', 'color')}): ").strip()
                else:
                    print(ch2)
                    v = ch2
                if v:
                    try:
                        idx = int(v) - 1
                        if 0 <= idx < len(choices):
                            state['output_color'] = choices[idx]
                    except Exception:
                        print("Invalid selection.")
                continue

            if k == '3':
                print("\n[Brightness & Contrast]")
                while True:
                    b = input(f"Brightness % (current: {state.get('brightness_percent', BRIGHTNESS_DEFAULT)}): ").strip()
                    if b == '':
                        break
                    try:
                        bp = float(b)
                        state['brightness_percent'] = bp
                        break
                    except Exception:
                        print("Invalid float.")
                while True:
                    c = input(f"Contrast % (current: {state.get('contrast_percent', state.get('contrast_level', CONTRAST_DEFAULT))}): ").strip()
                    if c == '':
                        break
                    try:
                        cp = float(c)
                        state['contrast_percent'] = cp
                        break
                    except Exception:
                        print("Invalid float.")
                continue

            if k == '4':
                state['card_only'] = not state.get('card_only', False)
                print("Card-only set to:", state['card_only'])
                time.sleep(0.2)
                continue

            if k == '5':
                array_submenu()
                continue

            if k == '6':
                v = input(f"Enter number of PDF copies (current: {state.get('pdf') or state.get('pdf_copies')}), blank to clear: ").strip()
                if v == '':
                    state['pdf'] = None
                    state['pdf_copies'] = None
                else:
                    try:
                        n = int(v)
                        if n <= 0:
                            raise ValueError()
                        state['pdf'] = n
                        state['pdf_copies'] = n
                    except Exception:
                        print("Invalid integer.")
                continue

            if k == '7':
                choices = ['png', 'jpg']
                print("1) png (lossless)  2) jpg (compressed)")
                sel = read_single_key()
                if sel == '':
                    sel = input("Choose (1-2): ").strip()
                else:
                    print(sel)
                if sel == '1':
                    state['image_format'] = 'png'
                elif sel == '2':
                    state['image_format'] = 'jpg'
                else:
                    print("Invalid selection.")
                time.sleep(0.2)
                continue

            print("Unknown selection in Output Options.")
            time.sleep(0.2)

    # Array submenu
    def array_submenu():
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("=== Array Options ===")
            print("1) Array mode    :", state.get('array'))
            print("   Description: 'separate' creates front/back pages; 'mixed' interleaves cards.")
            print("2) Grid spec     :", grid_spec_display(state))
            print("   Description: Specify rows cols using integers or 'max' token.")
            print("3) Mixed pattern :", state.get('mixed_pattern', MIXED_PATTERN_DEFAULT))
            print("   Description: Pattern when using mixed mode (top/left/right/bottom/checker).")
            print("r) Reset grid to auto")
            print("   Description: Clear grid override and use maximum fit.")
            print()
            print("Enter empty to return to Output Options.")
            sys.stdout.write("Choose (single key): ")
            sys.stdout.flush()
            ch = read_single_key()
            if ch == '':
                try:
                    line = input()
                except Exception:
                    line = ''
                if line.strip() == '':
                    return
                ch = line.strip()[0]
            if ch in ('\r', '\n'):
                return
            k = ch.lower()

            if k == '1':
                print("Array mode choices: 1) none  2) separate  3) mixed")
                sel = read_single_key()
                if sel == '':
                    sel = input("Choose (1-3): ").strip()
                else:
                    print(sel)
                choices = {'1': None, '2': 'separate', '3': 'mixed'}
                if sel in choices:
                    state['array'] = choices[sel]
                    print("Array mode set to:", state['array'])
                else:
                    print("Invalid choice.")
                time.sleep(0.2)
                continue

            if k == '2':
                max_r, max_c = compute_max_grid_for_state(state)
                print(f"Page capacity: {max_r} rows x {max_c} cols")
                print("Enter grid tokens like 'max', '3', or '3 4'. Leave empty to keep current.")
                print("Enter 'r' to reset to auto (max).")
                v = input(f"Grid (current: {state.get('grid_spec')}): ").strip()
                if v.lower() == 'r':
                    state['grid_spec'] = None
                    print("Grid reset to auto.")
                elif v:
                    toks = v.split()
                    try:
                        if len(toks) == 1:
                            if toks[0].lower() == 'max':
                                state['grid_spec'] = ['max']
                            else:
                                r = int(toks[0])
                                state['grid_spec'] = [str(r)]
                        else:
                            r_tok = toks[0].lower()
                            c_tok = toks[1].lower()
                            if r_tok != 'max':
                                int(r_tok)
                            if c_tok != 'max':
                                int(c_tok)
                            state['grid_spec'] = [r_tok, c_tok]
                    except Exception:
                        print("Invalid grid tokens; must be integers or 'max'.")
                continue

            if k == '3':
                print("Mixed pattern choices:")
                choices = ['top', 'left', 'right', 'checker', 'bottom']
                for i, p in enumerate(choices, start=1):
                    print(f" {i}) {p}")
                sel = read_single_key()
                if sel == '':
                    sel = input("Choose pattern (1-5): ").strip()
                else:
                    print(sel)
                try:
                    idx = int(sel) - 1
                    if 0 <= idx < len(choices):
                        state['mixed_pattern'] = choices[idx]
                        print("Mixed pattern set to:", state['mixed_pattern'])
                    else:
                        print("Invalid selection.")
                except Exception:
                    print("Invalid selection.")
                time.sleep(0.2)
                continue

            if k == 'r':
                state['grid_spec'] = None
                print("Grid reset to auto.")
                time.sleep(0.2)
                continue

            print("Unknown selection in Array Options.")
            time.sleep(0.2)

    # Help menu
    def help_menu():
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("=== Help / Manual ===")
            print(" - Use the menu to configure important parameters before selecting images.")
            print(" - Page Setup, Transform, and Output Options group related settings.")
            print(" - Front/Back images can be provided on CLI or selected after you accept the menu.")
            print()
            print("1) View quick help (this screen)")
            print("2) View full command-line help (-h)")
            print()
            print("Press Enter to return to main menu.")
            sys.stdout.write("Choose (single key): ")
            sys.stdout.flush()
            ch = read_single_key()
            if ch == '':
                try:
                    line = input()
                except Exception:
                    line = ''
                if line.strip() == '':
                    return
                ch = line.strip()[0]
            if ch in ('\r', '\n'):
                return
            if ch == '1':
                input("Quick help shown. Press Enter to go back.")
                return
            if ch == '2':
                if parser:
                    print("\nFull -h output:\n")
                    parser.print_help()
                    input("\nPress Enter to return to Help menu...")
                    continue
                else:
                    print("No parser available to show full -h.")
                    time.sleep(1)
                    continue
            return

    # Print CLI equivalent
    def print_cli_equivalent_menu():
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=== CLI Equivalent ===")
        cmd = build_cli_equivalent(state)
        print("\n" + cmd + "\n")
        input("Press Enter to return to menu...")

    # Page Setup submenu
    def page_setup_menu():
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            dims = get_paper_dimensions(state.get('paper'))
            print("=== Page Setup ===")
            print("1) Paper size   :", state.get('paper'), f"({dims[0]}mm x {dims[1]}mm)")
            print("   Description: Choose physical paper size used for output.")
            print("2) Orientation  :", state.get('orientation'))
            print("   Description: portrait or landscape. Changes page dimensions.")
            print()
            print("Enter empty to return to main menu.")
            sys.stdout.write("Choose (single key): ")
            sys.stdout.flush()
            ch = read_single_key()
            if ch == '':
                try:
                    line = input()
                except Exception:
                    line = ''
                if line.strip() == '':
                    return
                ch = line.strip()[0]
            if ch in ('\r', '\n'):
                return
            k = ch.lower()

            if k == '1':
                choices = list(PAPER_SIZES.items())
                for i, (name, dims) in enumerate(choices, start=1):
                    print(f" {i}) {name} - {dims[0]}mm x {dims[1]}mm")
                v = input(f"Choose paper (1-{len(choices)}) or leave empty to keep (current: {state.get('paper')}): ").strip()
                if v:
                    try:
                        idx = int(v) - 1
                        if 0 <= idx < len(choices):
                            state['paper'] = choices[idx][0]
                    except Exception:
                        print("Invalid choice.")
                continue

            if k == '2':
                print("\nOrientation: 1) portrait  2) landscape")
                sel = read_single_key()
                if sel == '':
                    sel = input("Choose (1-2): ").strip()
                else:
                    print(sel)
                if sel == '1':
                    state['orientation'] = 'portrait'
                elif sel == '2':
                    state['orientation'] = 'landscape'
                else:
                    print("Invalid selection.")
                time.sleep(0.2)
                continue

            print("Unknown selection in Page Setup.")
            time.sleep(0.2)

    # Main menu
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=== Configuration Menu ===")
        print("Press Enter (empty) to accept and continue, or 'q' to quit.")
        print()
        # Compact single-line entries, values aligned
        menu_items = [
            ("1) Front image", state.get('front') or "(none)"),
            ("2) Back image", state.get('back') or "(none)"),
            ("3) Output base", state.get('output_base')),
            ("4) Transform", f"Orientation={state.get('orientation')}, Align={state.get('align') or 'center'}"),
            ("5) Output Options", f"DPI={state.get('dpi')}, Color={state.get('output_color', state.get('output_mode', 'color'))}"),
            ("6) Page Setup", f"Paper={state.get('paper')}, Ori={state.get('orientation')}"),
            ("7) Print CLI command", ""),
            ("8) Help / Manual", ""),
            ("9) Quit", ""),
        ]
        # compute padding so values start aligned
        label_width = max(len(label) for label, _ in menu_items) + 2
        for label, val in menu_items:
            print(f"{label:<{label_width}}: {val}")
        print()
        sys.stdout.write("Select option (single key) or press Enter to accept: ")
        sys.stdout.flush()
        key = read_single_key()
        if key == '':
            try:
                line = input()
            except Exception:
                line = ''
            if line.strip() == '':
                return state
            key = line.strip()[0]

        if key in ('\r', '\n'):
            return state

        k = key.lower()

        if k == 'q' or k == '9':
            print("\nQuitting per user request.")
            sys.exit(0)

        if k == '1':
            res = file_selection_flow('front')
            if res is not None:
                state['front'] = res
            continue

        if k == '2':
            res = file_selection_flow('back')
            if res is not None:
                state['back'] = res
            continue

        if k == '3':
            v = input(f"New output base (current: {state.get('output_base')}): ").strip()
            if v:
                state['output_base'] = os.path.splitext(v)[0]
            continue

        if k == '4':
            transform_menu()
            continue

        if k == '5':
            output_options_menu()
            continue

        if k == '6':
            page_setup_menu()
            continue

        if k == '7':
            print_cli_equivalent_menu()
            continue

        if k == '8' or key == '?':
            help_menu()
            continue

        print("Unknown selection:", key)
        time.sleep(0.2)


def main():
    script_path = os.path.abspath(sys.argv[0])
    parser = build_arg_parser(script_path)
    args = parser.parse_args()

    # Prepare initial state from args
    state = {
        'front': args.front,
        'back': args.back,
        'output_base': os.path.splitext(args.output)[0],
        'dpi': args.dpi if hasattr(args, 'dpi') else 300,
        'output_color': getattr(args, 'output_color', getattr(args, 'output_mode', 'color')),
        'contrast_percent': getattr(args, 'contrast_percent', CONTRAST_DEFAULT),
        'brightness_percent': getattr(args, 'brightness_percent', BRIGHTNESS_DEFAULT),
        'no_crop': args.no_crop,
        'card_only': args.card_only,
        'array': args.array,
        'grid_spec': args.grid if hasattr(args, 'grid') else None,
        'paper': args.paper,
        'orientation': args.orientation,
        'pdf': args.pdf,
        'pdf_copies': getattr(args, 'pdf_copies', None),
        'align': None,
        'align_margin_mm': args.align_margin_mm if hasattr(args, 'align_margin_mm') else 10.0,
        'invert_h_flag': args.invert_align_h,
        'invert_v_flag': args.invert_align_v,
        'corner_area_pct': args.corner_area_pct if hasattr(args, 'corner_area_pct') else CORNER_AREA_PCT_DEFAULT,
        'rotate_map': parse_rotation_tokens(args.rotate if hasattr(args, 'rotate') else None),
        'gui_file_select': not args.no_gui_select,
        'mixed_pattern': args.mixed_pattern if hasattr(args, 'mixed_pattern') else MIXED_PATTERN_DEFAULT,
        'image_format': getattr(args, 'image_format', IMAGE_FORMAT_DEFAULT),
    }

    # If args.align provided, convert to textual align representation
    try:
        if args.align:
            ah, av = validate_align_tokens(args.align)
            state['align'] = align_string_from_parts(ah, av)
    except Exception:
        state['align'] = None

    # Decide whether to show interactive menu:
    show_menu_flag = False
    if sys.stdin and sys.stdin.isatty():
        if args.menu:
            show_menu_flag = True
        else:
            if not (args.front and args.back):
                show_menu_flag = True

    if show_menu_flag:
        state = interactive_config_review(state, parser=parser)

    # Update local variables from state
    front_path = state.get('front')
    back_path = state.get('back')
    output_base = state.get('output_base')
    dpi = state.get('dpi')
    args.dpi = dpi
    args.output_color = state.get('output_color')
    args.contrast_percent = state.get('contrast_percent')
    args.brightness_percent = state.get('brightness_percent')
    args.no_crop = state.get('no_crop')
    args.card_only = state.get('card_only')
    args.array = state.get('array')
    args.grid = state.get('grid_spec')
    args.paper = state.get('paper')
    args.orientation = state.get('orientation')
    args.pdf = state.get('pdf') or state.get('pdf_copies')
    args.image_format = state.get('image_format', IMAGE_FORMAT_DEFAULT)
    align_str = state.get('align')
    args.align_margin_mm = state.get('align_margin_mm')
    invert_h_flag = state.get('invert_h_flag')
    invert_v_flag = state.get('invert_v_flag')
    corner_area_pct = state.get('corner_area_pct')
    rotate_map = state.get('rotate_map', {})
    gui_file_select = state.get('gui_file_select', True)
    mixed_pattern = state.get('mixed_pattern', MIXED_PATTERN_DEFAULT)

    print("=" * 60)
    print("ID CARD SCANNER - Starting with configured options")
    print("=" * 60)

    # After menu: if front/back missing, request them now (respecting GUI toggle)
    if not front_path or not back_path:
        print("\nImage selection step (after menu):")
        if gui_file_select and (TK_AVAILABLE or shutil.which("zenity") or shutil.which("kdialog") or sys.platform == "darwin" or os.name == 'nt'):
            if not front_path:
                print("Please select FRONT image using file chooser...")
                sel = ask_file_gui("Select FRONT side image")
                if sel:
                    front_path = sel
            if not back_path:
                print("Please select BACK image using file chooser...")
                sel = ask_file_gui("Select BACK side image")
                if sel:
                    back_path = sel
        else:
            if not front_path:
                front_path = input("Enter path to FRONT side image: ").strip().strip('"').strip("'")
            if not back_path:
                back_path = input("Enter path to BACK side image: ").strip().strip('"').strip("'")

    if not front_path or not os.path.exists(front_path):
        print(f"Error: Front image not found or not provided: {front_path}")
        return

    if not back_path or not os.path.exists(back_path):
        print(f"Error: Back image not found or not provided: {back_path}")
        return

    front_img = cv2.imread(front_path)
    back_img = cv2.imread(back_path)

    if front_img is None or back_img is None:
        print("Error: Failed to load images!")
        return

    # Image rotations precedence
    rotate_image_front = None
    rotate_image_back = None
    if 'image-front' in rotate_map:
        rotate_image_front = rotate_map['image-front']
    if 'image-back' in rotate_map:
        rotate_image_back = rotate_map['image-back']
    if 'image' in rotate_map:
        if rotate_image_front is None:
            rotate_image_front = rotate_map['image']
        if rotate_image_back is None:
            rotate_image_back = rotate_map['image']

    rotate_image_front = 0 if rotate_image_front is None else int(rotate_image_front)
    rotate_image_back = 0 if rotate_image_back is None else int(rotate_image_back)
    if rotate_image_front not in (0, 90, 180, 270):
        print(f"Invalid rotate-image-front value: {rotate_image_front}")
        return
    if rotate_image_back not in (0, 90, 180, 270):
        print(f"Invalid rotate-image-back value: {rotate_image_back}")
        return

    if rotate_image_front != 0:
        front_img = rotate_image(front_img, rotate_image_front)
        print(f"Applied rotation {rotate_image_front}° to FRONT image before selection")
    if rotate_image_back != 0:
        back_img = rotate_image(back_img, rotate_image_back)
        print(f"Applied rotation {rotate_image_back}° to BACK image before selection")

    # Card and page sizing
    card_width_mm = 85.6
    card_height_mm = 53.98
    dpi = args.dpi if hasattr(args, 'dpi') else 300
    card_width_px = int((card_width_mm / 25.4) * dpi)
    card_height_px = int((card_height_mm / 25.4) * dpi)
    paper_width_mm, paper_height_mm = get_paper_dimensions(args.paper, args.orientation)
    page_width_px = int((paper_width_mm / 25.4) * dpi)
    page_height_px = int((paper_height_mm / 25.4) * dpi)
    align_margin_px = int((args.align_margin_mm / 25.4) * dpi) if hasattr(args, 'align_margin_mm') else int((10.0 / 25.4) * dpi)

    print(f"\nPaper size: {args.paper.upper()} ({paper_width_mm}mm x {paper_height_mm}mm) - {args.orientation}")
    print(f"DPI: {dpi}")
    print(f"Rotate image (front/back): {rotate_image_front} / {rotate_image_back} degrees")

    # Convert align string to parts for later layout application
    align_h, align_v = parse_align_string(align_str)

    if align_str:
        print(f"Requested alignment: {align_str} (h={align_h}, v={align_v}), margin={args.align_margin_mm} mm")
    if invert_h_flag or invert_v_flag:
        print(f"Invert Alignment every page - Horizontal: {invert_h_flag}, Vertical: {invert_v_flag}")
        if not (align_h or align_v):
            print("Warning: invert flags provided but align not set. Invert flags will be ignored.")

    array_mode = args.array
    array_rows = None
    array_cols = None
    grid_spec = args.grid if hasattr(args, 'grid') else None

    # Compute maximum grid that fits
    max_rows, max_cols = calculate_max_grid(card_width_px, card_height_px, page_width_px, page_height_px)

    # Parse grid option
    if array_mode:
        if grid_spec:
            if len(grid_spec) == 1:
                token = grid_spec[0].strip().lower()
                if token == 'max':
                    rows_token = None
                    cols_token = None
                else:
                    try:
                        v = int(token)
                        if v <= 0:
                            raise ValueError()
                        rows_token = v
                        cols_token = v
                    except Exception:
                        print(f"Grid parsing error: invalid token '{grid_spec[0]}'. Must be integer or 'max'.")
                        return
            else:
                try:
                    rows_token = parse_grid_token(grid_spec[0], "rows")
                    cols_token = parse_grid_token(grid_spec[1], "cols")
                except ValueError as e:
                    print(f"Grid parsing error: {e}")
                    return

            if rows_token is None:
                array_rows = max_rows
            else:
                if rows_token > max_rows:
                    msg = f"Requested rows {rows_token} exceeds page capacity ({max_rows}). Set rows to {max_rows}?"
                    if sys.stdin and sys.stdin.isatty():
                        keep = prompt_yes_no(msg, default_yes=True)
                        if not keep:
                            print("Aborting as requested.")
                            return
                        array_rows = max_rows
                    else:
                        array_rows = max_rows
                        print(f"Non-interactive: rows clamped to {array_rows}")
                else:
                    array_rows = rows_token

            if cols_token is None:
                array_cols = max_cols
            else:
                if cols_token > max_cols:
                    msg = f"Requested cols {cols_token} exceeds page capacity ({max_cols}). Set cols to {max_cols}?"
                    if sys.stdin and sys.stdin.isatty():
                        keep = prompt_yes_no(msg, default_yes=True)
                        if not keep:
                            print("Aborting as requested.")
                            return
                        array_cols = max_cols
                    else:
                        array_cols = max_cols
                        print(f"Non-interactive: cols clamped to {array_cols}")
                else:
                    array_cols = cols_token
        else:
            array_rows, array_cols = max_rows, max_cols
            print(f"Auto-calculated grid: {array_rows} rows x {array_cols} columns")

        array_rows = min(array_rows, max_rows)
        array_cols = min(array_cols, max_cols)

        print(f"Using grid: {array_rows} rows x {array_cols} columns (page capacity: {max_rows}x{max_cols})")

    # Determine effective alignment only if grid smaller than max on that axis
    effective_align_h = None
    effective_align_v = None
    if array_mode:
        if align_h:
            if array_cols < max_cols:
                effective_align_h = align_h
            else:
                print(f"Ignoring horizontal alignment '{align_h}' because columns ({array_cols}) use the page maximum ({max_cols}).")
        if align_v:
            if array_rows < max_rows:
                effective_align_v = align_v
            else:
                print(f"Ignoring vertical alignment '{align_v}' because rows ({array_rows}) use the page maximum ({max_rows}).")

    # Cropping / selection
    if args.no_crop:
        print("\nSkipping crop selection, using entire images...")
        front_card = resize_to_card_dimensions(front_img, card_width_px, card_height_px)
        back_card = resize_to_card_dimensions(back_img, card_width_px, card_height_px)
    else:
        print("\n" + "=" * 60)
        selector_front = CornerSelector(front_img, "Front Side - Select Corners", corner_area_pct=corner_area_pct)
        front_corners = selector_front.select()
        if front_corners is None:
            print("Front side selection cancelled.")
            return

        print("\n" + "=" * 60)
        selector_back = CornerSelector(back_img, "Back Side - Select Corners", corner_area_pct=corner_area_pct)
        back_corners = selector_back.select()
        if back_corners is None:
            print("Back side selection cancelled.")
            return

        print("\n" + "=" * 60)
        print("Processing images...")
        front_card = perspective_transform(front_img, front_corners, card_width_px, card_height_px)
        back_card = perspective_transform(back_img, back_corners, card_width_px, card_height_px)

    # Apply brightness/contrast + output color
    front_card = apply_output_color(front_card, args.output_color, contrast_percent=args.contrast_percent, brightness_percent=args.brightness_percent)
    back_card = apply_output_color(back_card, args.output_color, contrast_percent=args.contrast_percent, brightness_percent=args.brightness_percent)

    # Page rotations (after layout)
    rotate_page_front = None
    rotate_page_back = None
    if 'page-front' in rotate_map:
        rotate_page_front = rotate_map['page-front']
    if 'page-back' in rotate_map:
        rotate_page_back = rotate_map['page-back']
    if 'page' in rotate_map:
        if rotate_page_front is None:
            rotate_page_front = rotate_map['page']
        if rotate_page_back is None:
            rotate_page_back = rotate_map['page']

    rotate_page_front = 0 if rotate_page_front is None else int(rotate_page_front)
    rotate_page_back = 0 if rotate_page_back is None else int(rotate_page_back)
    if rotate_page_front not in (0, 90, 180, 270):
        print(f"Invalid rotate-page-front value: {rotate_page_front}")
        return
    if rotate_page_back not in (0, 90, 180, 270):
        print(f"Invalid rotate-page-back value: {rotate_page_back}")
        return

    print(f"Rotate pages (front/back): {rotate_page_front} / {rotate_page_back} degrees (applied after layout)")

    saved_paths = []
    output_folder = os.path.dirname(os.path.abspath(output_base)) or os.getcwd()

    def page_alignment_for_index(page_index, base_align_h, base_align_v):
        ah = base_align_h
        av = base_align_v
        if invert_h_flag and ah is not None and (page_index % 2 == 1):
            ah = flip_align(ah)
        if invert_v_flag and av is not None and (page_index % 2 == 1):
            av = flip_align(av)
        return ah, av

    def apply_page_rotation_to_image(img, is_front_page=True):
        deg = rotate_page_front if is_front_page else rotate_page_back
        if deg and deg in (90, 180, 270):
            return rotate_image(img, deg)
        return img

    # Card-only export (always PNG to preserve transparency)
    if args.card_only:
        print("Creating card-only exports with transparent background...")
        front_card_png = add_rounded_corners(front_card, radius_mm=3.18, dpi=dpi, transparent=True)
        back_card_png = add_rounded_corners(back_card, radius_mm=3.18, dpi=dpi, transparent=True)
        front_output = f"{output_base}_front.png"
        back_output = f"{output_base}_back.png"
        front_card_png.save(front_output)
        back_card_png.save(back_output)
        saved_paths.extend([os.path.abspath(front_output), os.path.abspath(back_output)])
        print(f"\n✓ Front card saved to: {front_output}")
        print(f"✓ Back card saved to: {back_output}")
        print(f"✓ Resolution: {dpi} DPI")
        print(f"✓ Card dimensions: {card_width_mm}mm x {card_height_mm}mm")
        print(f"✓ Format: PNG with transparent background")
        print("\n✓ Success! Card-only exports saved successfully!")
        print("=" * 60)
        if not args.no_open_after:
            if len(saved_paths) == 1:
                print(f"Opening exported file: {saved_paths[0]}")
                open_path(saved_paths[0])
            else:
                if OPEN_OUTPUT_FOLDER_DEFAULT and os.path.isdir(output_folder):
                    print(f"Opening export folder: {output_folder}")
                    open_path(output_folder)
        return

    # Add rounded corners non-transparent
    print("Adding rounded corners...")
    front_card = add_rounded_corners(front_card, radius_mm=3.18, dpi=dpi, transparent=False)
    back_card = add_rounded_corners(back_card, radius_mm=3.18, dpi=dpi, transparent=False)

    # Helper to save image file with chosen format
    def write_image(path, img, image_format):
        if image_format == 'jpg':
            # JPEG quality
            params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            return cv2.imwrite(path, img, params)
        else:
            # PNG default
            return cv2.imwrite(path, img)

    # Layout & export
    if array_mode == 'separate':
        print(f"Creating array layouts (separate pages: {array_rows}x{array_cols})...")
        front_cards = [front_card] * (array_rows * array_cols)
        back_cards = [back_card] * (array_rows * array_cols)

        if args.pdf:
            all_pages = []
            for copy_idx in range(args.pdf):
                page_index_front = copy_idx * 2
                page_index_back = copy_idx * 2 + 1
                ah_front, av_front = page_alignment_for_index(page_index_front, effective_align_h, effective_align_v)
                ah_back, av_back = page_alignment_for_index(page_index_back, effective_align_h, effective_align_v)
                page_front = create_array_layout(front_cards, array_rows, array_cols, page_width_px, page_height_px,
                                                 align_h=ah_front, align_v=av_front, margin_px=align_margin_px)
                page_back = create_array_layout(back_cards, array_rows, array_cols, page_width_px, page_height_px,
                                                align_h=ah_back, align_v=av_back, margin_px=align_margin_px)
                page_front = apply_page_rotation_to_image(page_front, is_front_page=True)
                page_back = apply_page_rotation_to_image(page_back, is_front_page=False)
                all_pages.extend([page_front, page_back])
            pdf_path = f"{output_base}.pdf"
            if save_as_pdf(all_pages, pdf_path, dpi):
                print(f"\n✓ PDF saved to: {pdf_path}")
                saved_paths.append(os.path.abspath(pdf_path))
                print(f"✓ Pages: {len(all_pages)} ({args.pdf} copies of 2-page layout)")
            else:
                print("\n✗ Failed to save PDF")
        else:
            ah_front, av_front = page_alignment_for_index(0, effective_align_h, effective_align_v)
            ah_back, av_back = page_alignment_for_index(1, effective_align_h, effective_align_v)
            page_front = create_array_layout(front_cards, array_rows, array_cols, page_width_px, page_height_px,
                                             align_h=ah_front, align_v=av_front, margin_px=align_margin_px)
            page_back = create_array_layout(back_cards, array_rows, array_cols, page_width_px, page_height_px,
                                            align_h=ah_back, align_v=av_back, margin_px=align_margin_px)
            page_front = apply_page_rotation_to_image(page_front, is_front_page=True)
            page_back = apply_page_rotation_to_image(page_back, is_front_page=False)
            ext = args.image_format or IMAGE_FORMAT_DEFAULT
            front_output = f"{output_base}_front.{ext}"
            back_output = f"{output_base}_back.{ext}"
            write_image(front_output, page_front, ext)
            write_image(back_output, page_back, ext)
            saved_paths.extend([os.path.abspath(front_output), os.path.abspath(back_output)])
            print(f"\n✓ Front page saved to: {front_output}")
            print(f"✓ Back page saved to: {back_output}")

        print(f"✓ Resolution: {dpi} DPI")
        print(f"✓ Grid: {array_rows} rows x {array_cols} columns")

    elif array_mode == 'mixed':
        print(f"Creating mixed array layout ({array_rows}x{array_cols}) pattern={mixed_pattern}...")
        if args.pdf:
            all_pages = []
            for copy_idx in range(args.pdf):
                page_index = copy_idx
                ah_page, av_page = page_alignment_for_index(page_index, effective_align_h, effective_align_v)
                page_mixed = create_mixed_array_layout(front_card, back_card, array_rows, array_cols,
                                                      page_width_px, page_height_px,
                                                      pattern=mixed_pattern,
                                                      align_h=ah_page, align_v=av_page, margin_px=align_margin_px)
                page_mixed = apply_page_rotation_to_image(page_mixed, is_front_page=True)
                all_pages.append(page_mixed)
            pdf_path = f"{output_base}.pdf"
            if save_as_pdf(all_pages, pdf_path, dpi):
                print(f"\n✓ PDF saved to: {pdf_path}")
                saved_paths.append(os.path.abspath(pdf_path))
                print(f"✓ Pages: {len(all_pages)} ({args.pdf} copies)")
            else:
                print("\n✗ Failed to save PDF")
        else:
            ah_page, av_page = page_alignment_for_index(0, effective_align_h, effective_align_v)
            page_mixed = create_mixed_array_layout(front_card, back_card, array_rows, array_cols,
                                                  page_width_px, page_height_px,
                                                  pattern=mixed_pattern,
                                                  align_h=ah_page, align_v=av_page, margin_px=align_margin_px)
            page_mixed = apply_page_rotation_to_image(page_mixed, is_front_page=True)
            ext = args.image_format or IMAGE_FORMAT_DEFAULT
            output_file = f"{output_base}.{ext}"
            write_image(output_file, page_mixed, ext)
            saved_paths.append(os.path.abspath(output_file))
            print(f"\n✓ Output saved to: {output_file}")

        print(f"✓ Resolution: {dpi} DPI")
        print(f"✓ Grid: {array_rows} rows x {array_cols} columns")
        print(f"✓ Total cards: {array_rows * array_cols} (mixed)")

    else:
        print("Creating standard layout...")
        page_standard = create_a4_layout(front_card, back_card, page_width_px, page_height_px)
        page_standard = apply_page_rotation_to_image(page_standard, is_front_page=True)

        if args.pdf:
            all_pages = [page_standard] * args.pdf
            pdf_path = f"{output_base}.pdf"
            if save_as_pdf(all_pages, pdf_path, dpi):
                print(f"\n✓ PDF saved to: {pdf_path}")
                saved_paths.append(os.path.abspath(pdf_path))
                print(f"✓ Pages: {len(all_pages)} ({args.pdf} copies)")
            else:
                print("\n✗ Failed to save PDF")
        else:
            ext = args.image_format or IMAGE_FORMAT_DEFAULT
            output_file = f"{output_base}.{ext}"
            write_image(output_file, page_standard, ext)
            saved_paths.append(os.path.abspath(output_file))
            print(f"\n✓ Output saved to: {output_file}")

        print(f"✓ Resolution: {dpi} DPI")

    if not args.no_open_after:
        if len(saved_paths) == 1:
            print(f"Opening exported file: {saved_paths[0]}")
            open_path(saved_paths[0])
        else:
            if OPEN_OUTPUT_FOLDER_DEFAULT and os.path.isdir(output_folder):
                print(f"Opening export folder: {output_folder}")
                open_path(output_folder)

    print(f"✓ Card dimensions: {card_width_mm}mm x {card_height_mm}mm")
    print(f"✓ Page format: {args.paper.upper()} ({paper_width_mm}mm x {paper_height_mm}mm) - {args.orientation}")
    if args.pdf:
        print("✓ Output format: PDF")
    else:
        print(f"✓ Output format: {args.image_format if args.image_format else IMAGE_FORMAT_DEFAULT}")
    print("\n✓ Success! ID card layout saved successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
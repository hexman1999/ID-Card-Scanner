# ID Card Scanner

A command-line and interactive tool to extract ID/credit-card-sized images from photos and compose printable layouts (single pages or arrays). This README documents all CLI options, interactive menu features and behaviors, and implementation notes.

---

## Table of Contents

- Overview
- Requirements
- Quick Install
- Basic Usage
- Full CLI Options (all flags and arguments)
- Interactive Menu — Overview and behavior
- Submenus and descriptions
  - Page Setup
  - Transform
  - Output Options (including Array / Grid)
  - Array submenu (grid, mixed pattern)
  - Help and CLI-equivalent
- Corner selector (interactive) — controls & features
- Output behavior and formats
- Examples
- Notes & tips
- License

---

## Overview

This tool extracts card-sized rectangles (85.6 mm × 53.98 mm) from front/back photographs and produces printable layouts (PNG or PDF), with many options for alignment, grid layout and image processing (brightness/contrast, color modes). It supports both fully non-interactive CLI runs and an interactive TTY menu with GUI file selectors (when available).

Interactive behavior highlights:
- Compact main menu with single-key selection for most options.
- Submenus expose full descriptions.
- Corner selection GUI with a high-quality magnifier and keyboard pixel-step editing for precise corner placement.
- Menu can be skipped when both front and back paths are supplied on the CLI unless `-m|--menu` is passed.

---

## Requirements

- Python 3.8+
- OpenCV (cv2)
- numpy
- Pillow (PIL)
- Optional: tkinter or a native file selector (zenity/kdialog on Linux, AppleScript on macOS, PowerShell on Windows) for GUI file selection.

Install requirements e.g.:
```
pip install opencv-python numpy pillow
```
Install platform GUI helpers as needed (zenity/kdialog on Linux).

---

## Quick Install

Save the script (e.g., `id_card_scanner.py`), ensure it's executable:
```
chmod +x id_card_scanner.py
```

Run:
```
./id_card_scanner.py [front.jpg] [back.jpg] [options]
```

---

## Basic Usage

- Non-interactive example:
  - Provide both images and flags:
    ```
    ./id_card_scanner.py front.jpg back.jpg -o mycards --dpi 300 --output-color enhanced --image-format png --pdf 2
    ```
- Interactive example:
  - Run without both images (or with `-m`) to open interactive menu:
    ```
    ./id_card_scanner.py
    ```

---

## Full CLI Options

All CLI arguments accepted by the tool:

Positional
- `front` (optional) — path to front side image
- `back` (optional) — path to back side image

Output & processing
- `-o, --output` — Output file base path (no extension). Default: `id_card_output`.
- `--dpi` — DPI used to compute pixel sizes (integer). Default: `300`.
- `--output-color` — Output color mode. Choices:
  - `color` (default)
  - `grayscale`
  - `enhanced` (CLAHE + HSV saturation boost)
  - `blackwhite` (Otsu threshold)
- `--brightness-percent` — Brightness percent (100 = unchanged). Default: `100.0`
- `--contrast-percent` — Contrast percent (100 = unchanged). Default: `100.0`
- `--image-format` — Output image format for non-PDF output: `png` (lossless) or `jpg` (compressed). Default: `png`
  - Note: `--image-format` does not change `--card-only` (card-only always produces PNG to preserve transparency).

Layout & grids
- `--array` — `separate` or `mixed`. `separate` produces separate front/back pages; `mixed` interleaves cards in one page layout.
- `--grid` — Grid tokens: specify rows/cols as integers or `max`. Accepts one or two tokens:
  - `--grid 3` => 3×3 grid
  - `--grid max` => use maximum rows & cols that fit the page
  - `--grid max 3` => max rows, 3 cols
- `--mixed-pattern` — Pattern for mixed arrays: `top`, `left`, `right`, `checker`, `bottom`. Default: `top`.
- `--paper` — Paper name; choices: `a4`, `a3`, `a5`, `letter`, `legal`, `tabloid`. Default: `a4`.
- `--orientation` — `portrait` or `landscape`. Default: `portrait`.
- `--align` — One or two tokens to align grid when it does not fill the page. Allowed tokens: `left`, `right`, `top`, `bottom`. Example: `--align left top`.
- `--align-margin-mm` — Margin in millimeters used for alignment (default `10.0`)

PDF / copies
- `--pdf [N]` — Export PDF. Optional integer argument sets number of copies (if omitted, const=1). Example: `--pdf 2`.
- `--pdf-copies` — (alias/utility) set number of PDF copies (script supports both where present).

Selection, cropping & corner selector
- `--no-crop` — Skip cropping: use full image stretched to card size.
- `--corner-area-pct` — Fraction (0.0–1.0) controlling selection grip circle area. Default `0.20`.

File selection behavior
- `--no-gui-select` — Disable GUI file chooser; use CLI input only.

Rotation tokens
- `--rotate` — Accepts multiple tokens `key=deg` where key ∈ { `image`, `image-front`, `image-back`, `page`, `page-front`, `page-back` } and deg ∈ {0,90,180,270}. Examples:
  - `--rotate image=90 page=180`
  - `--rotate image-front=90`

Invert alignment toggles
- `--invert-align-h` — Alternate/invert horizontal alignment on alternating pages (left↔right).
- `--invert-align-v` — Alternate/invert vertical alignment on alternating pages (top↔bottom).

Other flags
- `--card-only` — Export card-only PNGs with rounded corners and transparency (front & back).
- `--no-open-after` — Do not open exported file/folder after finish.
- `--no-open-folder` — Do not open export folder after finish (CLI).
- `-m, --menu` — Force interactive menu even if both front/back provided.

---

## Interactive Menu — Behavior Summary

When invoked in a TTY without both positional images (or when `-m` is used), the interactive menu appears. Main points:

- Compact main menu: each entry shows a single-line summarized value beside the menu label (aligned).
- Most menu entries accept a single key (1..9, then letters for sub-entries). Pressing the shown key toggles/edits that option.
- Pressing Enter on an empty prompt accepts the current configuration and continues.
- Press `q` or choose Quit to exit the program immediately.

Menu groups:
- `Front image`, `Back image` — image selection flows:
  - If a path is set: show current path and allow `Enter` to keep, `c` to change, or `b`/ESC to go back to menu.
  - If changing or no current path: prompt for typed path (Enter to accept). Leave empty + Enter to open GUI file chooser (if available). Type `b` then Enter or press ESC to return to the menu without change.
- `Transform` — orientation, image/page rotation, invert alignment H/V, and Align (human readable like `top left`). Submenu includes descriptions for each item (see below).
- `Page Setup` — select paper and orientation. The submenu shows paper dimension values.
- `Output Options` — DPI, Output Color, Brightness/Contrast (percent values), Card-only toggle, Array submenu, PDF copies, Image format (png/jpg). Submenu shows descriptions and is the place to set grid & mixed-pattern.
- `Array` submenu — configure array mode (`separate`/`mixed`), edit `--grid`, choose `--mixed-pattern`, or reset grid to `auto`.

Help & CLI-equivalent:
- Help submenu provides a short manual and option to view the full `-h` (argparse) output.
- A "Print CLI command" entry prints the equivalent CLI command string for the current interactive settings so you can re-run non-interactively.

---

## Submenus & Descriptions

Descriptions are shown inside submenus (not cluttering main menu). Example details:

### Page Setup (submenu)
- Description: choose paper size and orientation. Paper sizes are listed with their physical dimensions in mm (e.g., `a4 - 210mm x 297mm`).

### Transform (submenu)
- Orientation: set `portrait` or `landscape`.
  - Description: changes page dimensions used to compute grid capacity.
- Image/Page Rotation:
  - Description: choose a scope (global image/page or per-side) and a preset rotation from {0, 90, 180, 270}. Rotations for images are applied before corner selection.
- Invert Alignment (Horizontal):
  - Description: flip left↔right alignment on alternating pages.
- Invert Alignment (Vertical):
  - Description: flip top↔bottom alignment on alternating pages.
- Align:
  - Description: choose alignment expressed in natural language (`top left`, `center`, `bottom right`), used when the grid does not use full page capacity.

### Output Options (submenu)
- DPI:
  - Description: DPI used for computing card and page pixel sizes.
- Output Color:
  - Description: Color processing mode (color/grayscale/enhanced/blackwhite).
- Brightness / Contrast:
  - Description: Percent values; 100% = unchanged.
- Card-only Export:
  - Description: Export card images only (PNG with transparency and rounded corners).
- Array:
  - Description: Opens the array submenu to configure grid and mixed-pattern.
- PDF copies:
  - Description: Number of PDF copies when saving to PDF (also controlled by `--pdf`).
- Image format:
  - Description: For non-PDF exports, choose `png` (lossless) or `jpg` (compressed). Card-only exports remain PNG.

### Array submenu (under Output Options)
- Array mode:
  - `none`, `separate`, `mixed`
  - Description: `separate` creates a separate front and back page; `mixed` places fronts/backs on a single page in a pattern.
- Grid spec:
  - Description: Enter tokens such as `max`, `3`, `3 4`. `max` resolves to the maximum rows/cols the page supports at the selected DPI/paper orientation.
  - Reset: `r` resets grid spec to `auto` (max).
- Mixed pattern:
  - Description: `top`, `left`, `right`, `checker`, `bottom`. Default: `top`.

---

## Corner Selector (interactive)

When cropping (not using `--no-crop`), a GUI window opens for front and back images with these features:

- Corner grips: 4 draggable points (TL, TR, BR, BL) that define a polygon for perspective transform.
- Magnifying glass:
  - High-quality magnifier overlay shows a high-resolution upscaled region around the active corner using LANCZOS (or adaptive linear for very large images).
  - The magnifier renders a soft shadow/backdrop and crisp white/black borders for visibility.
  - The magnifier uses an anti-aliased circular mask and blends with surrounding image for a polished appearance.
- Pan & zoom:
  - Middle mouse button drag to pan when zoomed.
  - Mouse wheel to zoom in/out.
  - `Z` to auto-zoom to the corners.
  - `A` to select entire image.
- Keyboard pixel editing:
  - While in the corner selector, you can move the last selected corner pixel-by-pixel using the arrow keys.
  - When you press any arrow key, the magnifier remains visible for 3 seconds after the last press to aid in fine adjustments.
  - This is useful for sub-pixel-precise placement (whole-pixel movement).
- Other controls:
  - `R` — Reset corners to sensible default.
  - `Enter` — Accept and continue.
  - `ESC` — Cancel selection (returns to main flow, which will abort processing if no selection).

Implementation notes:
- The corner-area selection circle size is computed from `--corner-area-pct` (fraction of image area). This controls the clickable drag radius and visual ring size.
- For very large images, the magnifier uses faster interpolation for the main view to keep the UI responsive, while the magnifier ROI uses high-quality upscaling for detail.

---

## Output Behavior & Formats

- Card-only mode:
  - Produces PNG files with rounded corners and transparent background (`<output>_front.png`, `<output>_back.png`).
- Page layouts:
  - Default single-page layout: front and back cards vertically stacked on an A4 page (or chosen paper).
  - `array` modes:
    - `separate`: creates two pages (front & back) at the chosen grid size.
    - `mixed`: produces a single page with interleaved cards according to `--mixed-pattern`.
  - PDF export:
    - Use `--pdf N` to export PDF(s). Multiple pages are appended and saved into `<output>.pdf`.
  - Non-PDF export:
    - Single-page PNG/JPG output based on `--image-format`.
- Rounding, quality:
  - Perspective extraction uses high-quality interpolation (LANCZOS) for best visual results.
  - When written to JPG, quality is set high (e.g., 95). PNG is lossless.

---

## Examples

- Interactive run (menu):
  ```
  ./id_card_scanner.py
  ```

- Non-interactive: produce two-page PDF (2 copies) using enhanced color processing:
  ```
  ./id_card_scanner.py front.jpg back.jpg -o mycards --dpi 300 --output-color enhanced --pdf 2
  ```

- Non-interactive: create a 3×4 PNG page using JPEG compressed output:
  ```
  ./id_card_scanner.py front.jpg back.jpg -o sheet --array separate --grid 3 4 --image-format jpg
  ```

- Force interactive menu even though images are provided:
  ```
  ./id_card_scanner.py front.jpg back.jpg -m
  ```

- Rotate front image 90 degrees before selection, and rotate page 180 degrees after layout:
  ```
  ./id_card_scanner.py front.jpg back.jpg --rotate image-front=90 page=180
  ```

---

## Notes & Tips

- If you wish to script repeated runs, use the `Print CLI command` option inside the interactive menu to get the equivalent one-line invocation you can reuse.
- If your environment lacks GUI file selectors, use `--no-gui-select` to force CLI-only path entry.
- When using `--image-format jpg` you will get compressed output; choose `png` for best fidelity.
- Card-only exports are always PNG to preserve transparency.
- The `--corner-area-pct` setting can be used to tune the drag sensitivity on particularly noisy images or when the card corners are small relative to the photo.

---

## Troubleshooting

- If OpenCV window does not render correctly on remote sessions, ensure you're running in a desktop environment with display access or use non-interactive mode.
- If images fail to load, ensure paths are correct and readable.
- If GUI file selector fails, install system helper (zenity/kdialog) or ensure `tkinter` is available.

---

## License

This README and the accompanying script are provided as-is. You may adapt and modify the script for your own use.


Note: this project was made using claude ai + github copilot guided by me (just for reference)

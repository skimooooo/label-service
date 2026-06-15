"""
Deterministic shipping-label compositor v4.

- High-resolution flat label (4x), downsampled with LANCZOS for crisp text
- Real QR code cropped from base photo, pasted back at original position
- TF-XXXXXXXXXX tracking format, barcode visually matches its digits
- Sampled paper color from base photo (off-white, not blue)
- Minimal expansion (0.5-1%) of destination corners
- Clean To: block (no phone unless provided), supports German ß etc.
- Very light blur/noise for realism
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BASE_IMAGE_PATH = "base.png"

SCALE = 4  # supersampling factor for crisp text
LABEL_W, LABEL_H = 600 * SCALE, 850 * SCALE

# Four corners of the real label in the base photo: TL, TR, BR, BL
DEST_CORNERS = np.array([
    [596, 172],
    [872, 181],
    [935, 494],
    [571, 490],
], dtype=np.float32)

# (QR region removed in this version - no paste-back)

FONT_DIR = "/usr/share/fonts/truetype/liberation"
FONT_BOLD = f"{FONT_DIR}/LiberationSans-Bold.ttf"
FONT_REG = f"{FONT_DIR}/LiberationSans-Regular.ttf"

TEXT_COLOR = (25, 25, 30)
LINE_COLOR_STRONG = (50, 50, 55)
LINE_COLOR_THIN = (90, 90, 95)


def get_paper_color(base_bgr: np.ndarray) -> tuple:
    """Sample median paper color from clean areas of the original label."""
    samples_xy = [(650, 200), (700, 400), (800, 470), (620, 350), (750, 480)]
    samples = []
    for x, y in samples_xy:
        b, g, r = base_bgr[y, x]
        samples.append((int(r), int(g), int(b)))
    arr = np.array(samples, dtype=float)
    median = np.median(arr, axis=0)
    # Pull slightly toward neutral gray (desaturate) for an off-white look
    gray = median.mean()
    desat = 0.25
    final = median * (1 - desat) + gray * desat
    return tuple(int(c) for c in final)


# ---------------------------------------------------------------------------
# Build the flat label with Pillow (high resolution)
# ---------------------------------------------------------------------------

def build_flat_label(data: dict, paper_color: tuple) -> Image.Image:
    """
    data keys: recipient_name, address_line_1, postal_city, country,
               phone (optional), tracking_number (format TF-XXXXXXXXXX)
    """
    s = SCALE
    label = Image.new("RGB", (LABEL_W, LABEL_H), paper_color)
    draw = ImageDraw.Draw(label)

    margin = 30 * s
    w = LABEL_W

    f_brand = ImageFont.truetype(FONT_BOLD, 38 * s)
    f_service_small = ImageFont.truetype(FONT_BOLD, 20 * s)
    f_route = ImageFont.truetype(FONT_BOLD, 46 * s)
    f_trk = ImageFont.truetype(FONT_BOLD, 26 * s)
    f_label_small = ImageFont.truetype(FONT_REG, 16 * s)
    f_text = ImageFont.truetype(FONT_REG, 17 * s)
    f_text_bold = ImageFont.truetype(FONT_BOLD, 18 * s)
    f_footer = ImageFont.truetype(FONT_REG, 14 * s)

    y = margin

    # --- Header: TrackFlow / TF-EXPRESS ---
    draw.text((margin, y), "TrackFlow", font=f_brand, fill=TEXT_COLOR)
    tf_text = "TF-EXPRESS"
    tf_w = draw.textlength(tf_text, font=f_service_small)
    draw.text((w - margin - tf_w, y + 14 * s), tf_text, font=f_service_small, fill=TEXT_COLOR)
    y += 60 * s
    draw.line([(margin, y), (w - margin, y)], fill=LINE_COLOR_STRONG, width=3 * s)
    y += 25 * s

    # --- Route ---
    route_text = "US - DE - DE"
    route_w = draw.textlength(route_text, font=f_route)
    draw.text(((w - route_w) / 2, y), route_text, font=f_route, fill=TEXT_COLOR)
    y += 75 * s
    draw.line([(margin, y), (w - margin, y)], fill=LINE_COLOR_THIN, width=2 * s)
    y += 20 * s

    # --- Barcode: visually correspond to the tracking number digits ---
    barcode_h = 55 * s
    digits = "".join(ch for ch in data["tracking_number"] if ch.isdigit())
    if not digits:
        digits = "0123456789"
    x = margin + 10 * s
    bx_end = w - margin - 10 * s
    avail = bx_end - x
    # derive bar widths from digits, cycling through them
    di = 0
    while x < bx_end:
        d = int(digits[di % len(digits)])
        di += 1
        bw = (2 + (d % 5)) * s  # bar width 2-6 px (scaled), driven by digit
        gap = (1 + ((d * 3) % 4)) * s
        if d % 7 != 0:  # mostly draw bars, occasional gap-only
            draw.rectangle([x, y, x + bw, y + barcode_h], fill=(15, 15, 15))
        x += bw + gap
    y += barcode_h + 12 * s

    # --- Tracking number ---
    trk_text = data["tracking_number"]
    trk_w = draw.textlength(trk_text, font=f_trk)
    draw.text(((w - trk_w) / 2, y), trk_text, font=f_trk, fill=TEXT_COLOR)
    y += 45 * s
    draw.line([(margin, y), (w - margin, y)], fill=LINE_COLOR_THIN, width=2 * s)
    y += 18 * s

    # --- From / To columns ---
    col_split = w // 2

    draw.text((margin, y), "From:", font=f_text_bold, fill=TEXT_COLOR)
    fy = y + 24 * s
    for line in [
        "Sunny Global Trading PLLC",
        "200 Pine St, 23rd",
        "Seattle, WA 98101",
        "United States",
        "+1 971-123 4567",
    ]:
        draw.text((margin, fy), line, font=f_text, fill=TEXT_COLOR)
        fy += 24 * s

    tx = col_split + 25 * s
    available_w = w - margin - tx

    def estimate_lines(text, font, avail_w):
        words = text.split(" ")
        lines = 1
        line = ""
        for word in words:
            test_line = (line + " " + word).strip()
            if draw.textlength(test_line, font=font) <= avail_w:
                line = test_line
            else:
                lines += 1
                line = word
        return lines

    # Build the To: field list - only include phone if provided
    to_fields = [
        (data["recipient_name"], f_text_bold, 26 * s),
        (data["address_line_1"], f_text, 24 * s),
        (data["postal_city"], f_text, 24 * s),
        (data["country"], f_text, 24 * s),
    ]
    if data.get("phone"):
        to_fields.append((data["phone"], f_text, 24 * s))

    total_lines = sum(estimate_lines(text, font, available_w) for text, font, _ in to_fields)

    # Shrink font if content would overflow
    if total_lines > 6:
        f_text_small = ImageFont.truetype(FONT_REG, 14 * s)
        f_text_bold_small = ImageFont.truetype(FONT_BOLD, 15 * s)
        to_fields = [
            (text, (f_text_bold_small if font is f_text_bold else f_text_small), int(lh * 20 / 24))
            for text, font, lh in to_fields
        ]

    draw.text((tx, y), "To:", font=f_text_bold, fill=TEXT_COLOR)
    ty = y + 24 * s

    def draw_wrapped(text, font, ty, line_h):
        words = text.split(" ")
        line = ""
        for word in words:
            test_line = (line + " " + word).strip()
            if draw.textlength(test_line, font=font) <= available_w:
                line = test_line
            else:
                draw.text((tx, ty), line, font=font, fill=TEXT_COLOR)
                ty += line_h
                line = word
        if line:
            draw.text((tx, ty), line, font=font, fill=TEXT_COLOR)
            ty += line_h
        return ty

    for text, font, line_h in to_fields:
        ty = draw_wrapped(text, font, ty, line_h)

    draw.line([(col_split, y - 5 * s), (col_split, max(fy, ty) + 5 * s)], fill=LINE_COLOR_THIN, width=2 * s)

    y = max(fy, ty) + 15 * s
    draw.line([(margin, y), (w - margin, y)], fill=LINE_COLOR_THIN, width=2 * s)
    y += 18 * s

    # --- Service / Weight / Pieces row ---
    col_w = (w - 2 * margin) / 3
    draw.text((margin, y), "Service", font=f_label_small, fill=(95, 95, 100))
    draw.text((margin, y + 22 * s), "EXPRESS", font=f_text_bold, fill=TEXT_COLOR)

    draw.text((margin + col_w, y), "Weight", font=f_label_small, fill=(95, 95, 100))
    draw.text((margin + col_w, y + 22 * s), "12.40 KG", font=f_text_bold, fill=TEXT_COLOR)

    draw.text((margin + 2 * col_w, y), "Pieces", font=f_label_small, fill=(95, 95, 100))
    draw.text((margin + 2 * col_w, y + 22 * s), "1 / 1", font=f_text_bold, fill=TEXT_COLOR)

    y += 60 * s
    draw.line([(margin, y), (w - margin, y)], fill=LINE_COLOR_THIN, width=2 * s)
    y += 18 * s

    # --- Parcel Ref (no QR drawn here - real QR pasted back after warp) ---
    draw.text((margin, y), "Parcel Ref:", font=f_text_bold, fill=TEXT_COLOR)
    draw.text((margin, y + 24 * s), "EGT-5400-78910", font=f_text_bold, fill=TEXT_COLOR)

    y += 70 * s
    draw.line([(margin, y), (w - margin, y)], fill=LINE_COLOR_THIN, width=2 * s)
    y += 18 * s

    # --- Footer ---
    draw.text((margin, y), "trackflow.ltd/track", font=f_footer, fill=(95, 95, 100))

    # Downsample to target resolution with high-quality filter before
    # warping (large ratio downsampling needs Lanczos pre-filtering to
    # avoid aliasing; the warp itself uses CUBIC to avoid ringing)
    final = label.resize((LABEL_W // SCALE, LABEL_H // SCALE), Image.LANCZOS)
    return final


# ---------------------------------------------------------------------------
# Warp flat label onto base photo
# ---------------------------------------------------------------------------

def warp_and_composite(base_img_bgr: np.ndarray, label_img: Image.Image, dest_corners: np.ndarray) -> np.ndarray:
    label_cv = cv2.cvtColor(np.array(label_img), cv2.COLOR_RGB2BGR)
    lh, lw = label_cv.shape[:2]

    src_corners = np.array([
        [0, 0],
        [lw, 0],
        [lw, lh],
        [0, lh],
    ], dtype=np.float32)

    h, w = base_img_bgr.shape[:2]

    # Minimal expansion (0.8%) just to cover old label edges
    centroid = dest_corners.mean(axis=0)
    expanded_corners = centroid + (dest_corners - centroid) * 1.008
    M = cv2.getPerspectiveTransform(src_corners, expanded_corners.astype(np.float32))
    warped_label = cv2.warpPerspective(label_cv, M, (w, h), flags=cv2.INTER_CUBIC)

    mask = np.ones((lh, lw), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_LINEAR)

    # Very light realism adjustments
    warped_label = cv2.GaussianBlur(warped_label, (3, 3), 0.3)
    noise = np.random.default_rng(3).normal(0, 1.2, warped_label.shape).astype(np.float32)
    warped_label = np.clip(warped_label.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    sharpen_kernel = np.array([[0, -0.04, 0], [-0.04, 1.16, -0.04], [0, -0.04, 0]])
    warped_label = cv2.filter2D(warped_label, -1, sharpen_kernel)

    warped_mask = cv2.GaussianBlur(warped_mask, (5, 5), 1.5)
    mask_f = (warped_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

    result = (base_img_bgr.astype(np.float32) * (1 - mask_f) + warped_label.astype(np.float32) * mask_f)
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Whole-image "real photo" degradation pass
# ---------------------------------------------------------------------------

def apply_photo_realism(img_bgr: np.ndarray) -> np.ndarray:
    """
    Apply subtle whole-image degradation so the final result looks like a
    real (slightly compressed) warehouse phone photo rather than a crisp
    digital render. Effects: tiny blur, slight micro-contrast reduction,
    a small downscale/upscale roundtrip, and JPEG re-encoding at moderate
    quality. Kept light enough that label text stays readable at normal
    viewing size.
    """
    h, w = img_bgr.shape[:2]

    # 1. Very light overall blur
    img = cv2.GaussianBlur(img_bgr, (3, 3), 0.5)

    # 2. Slight micro-contrast reduction (pull values toward mid-gray a touch)
    img_f = img.astype(np.float32)
    img_f = img_f * 0.97 + 128 * 0.03
    img = np.clip(img_f, 0, 255).astype(np.uint8)

    # 3. Small downscale/upscale roundtrip to soften fine detail slightly
    small = cv2.resize(img, (int(w * 0.92), int(h * 0.92)), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 4. JPEG re-encode at moderate quality to introduce compression artifacts
    encode_quality = 78
    success, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, encode_quality])
    if success:
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return img


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_label(recipient: dict, tracking_number: str = None, out_path: str = None):
    """
    recipient: dict with keys name, line1 (street), line2 (city), line3
               (postal code), line4 (country), phone (optional)
    tracking_number: string in format 'TF-XXXXXXXXXX' (digits). If a plain
                     number is passed, it will be formatted as TF-<digits>.
    Returns a PIL.Image.Image (RGB). If out_path is given, also saves to disk.
    """
    postal_code = recipient.get("line3", "")
    city = recipient.get("line2", "")
    if postal_code and city:
        postal_city_combined = f"{postal_code} {city}"
    else:
        postal_city_combined = postal_code or city

    # Normalize tracking number to TF-XXXXXXXXXX format
    trk = tracking_number or ""
    digits_only = "".join(ch for ch in trk if ch.isdigit())
    if trk.upper().startswith("TF-"):
        trk_formatted = trk.upper()
    elif digits_only:
        trk_formatted = f"TF-{digits_only}"
    else:
        trk_formatted = trk

    data = {
        "recipient_name": recipient["name"],
        "address_line_1": recipient.get("line1", ""),
        "postal_city": postal_city_combined,
        "country": recipient.get("line4", ""),
        "phone": recipient.get("phone", ""),
        "tracking_number": trk_formatted,
    }

    base_bgr = cv2.imread(BASE_IMAGE_PATH)
    if base_bgr is None:
        raise FileNotFoundError(BASE_IMAGE_PATH)

    paper_color = get_paper_color(base_bgr)
    flat = build_flat_label(data, paper_color)

    result_bgr = warp_and_composite(base_bgr, flat, DEST_CORNERS)

    # QR code intentionally removed - the flat label's Parcel Ref area
    # already has blank background + correct dividers, no paste-back needed.

    result_bgr = apply_photo_realism(result_bgr)

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    result_img = Image.fromarray(result_rgb)

    if out_path:
        result_img.save(out_path)
    return result_img


if __name__ == "__main__":
    img = generate_label(
        {
            "name": "Skylar Lippert",
            "line1": "Lise-Meitner-Straße 21",
            "line2": "Geilenkirchen",
            "line3": "52511",
            "line4": "Germany",
        },
        tracking_number="5839174268",
        out_path="test_v4.png",
    )
    print("done", img.size)

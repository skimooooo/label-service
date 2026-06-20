"""
Deterministic shipping-label compositor v5.

Main fix in v5:
- The barcode is NEVER generated and NEVER processed by blur/JPEG/noise.
- The original barcode region is copied from the base photo as the very last step.
- QR code is removed: no QR is drawn and no QR crop is pasted back.
- Final image can still get the low-quality phone-photo look, but the barcode stays static.

Use this only for mockups / internal previews, not as a real shipping label.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

# IMPORTANT:
# This region is copied from the base photo AFTER all blur/compression.
# Keep it tight enough to avoid old tracking text, but wide enough to cover
# the entire barcode and a small paper margin.
# Tune these numbers using your base.png if needed.
BARCODE_REGION = (615, 225, 860, 260)

FONT_DIR = "/usr/share/fonts/truetype/liberation"
FONT_BOLD = f"{FONT_DIR}/LiberationSans-Bold.ttf"
FONT_REG = f"{FONT_DIR}/LiberationSans-Regular.ttf"

TEXT_COLOR = (25, 25, 30)
LINE_COLOR_STRONG = (50, 50, 55)
LINE_COLOR_THIN = (90, 90, 95)


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    """Load font with fallback."""
    if os.path.exists(path):
        return ImageFont.truetype(path, size)

    fallback_paths = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "arial.ttf",
    ]
    for fp in fallback_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)

    return ImageFont.load_default()


def get_paper_color(base_bgr: np.ndarray) -> tuple:
    """Sample median paper color from clean areas of the original label."""
    samples_xy = [(650, 200), (700, 400), (800, 470), (620, 350), (750, 480)]
    samples = []

    h, w = base_bgr.shape[:2]
    for x, y in samples_xy:
        if 0 <= x < w and 0 <= y < h:
            b, g, r = base_bgr[y, x]
            samples.append((int(r), int(g), int(b)))

    if not samples:
        return (220, 226, 232)

    arr = np.array(samples, dtype=float)
    median = np.median(arr, axis=0)

    # Pull slightly toward neutral gray for off-white photographed paper.
    gray = median.mean()
    desat = 0.25
    final = median * (1 - desat) + gray * desat

    return tuple(int(c) for c in final)


# ---------------------------------------------------------------------------
# Build the flat label with Pillow
# ---------------------------------------------------------------------------

def build_flat_label(data: dict, paper_color: tuple) -> Image.Image:
    """
    data keys:
        recipient_name
        address_line_1
        postal_city
        country
        phone (optional)
        tracking_number, format TF-XXXXXXXXXX

    IMPORTANT:
    This function intentionally does NOT draw a barcode.
    The barcode is copied from the original photo at the very end.
    """
    s = SCALE
    label = Image.new("RGB", (LABEL_W, LABEL_H), paper_color)
    draw = ImageDraw.Draw(label)

    margin = 30 * s
    w = LABEL_W

    f_brand = _load_font(FONT_BOLD, 38 * s)
    f_service_small = _load_font(FONT_BOLD, 20 * s)
    f_route = _load_font(FONT_BOLD, 46 * s)
    f_trk = _load_font(FONT_BOLD, 26 * s)
    f_label_small = _load_font(FONT_REG, 16 * s)
    f_text = _load_font(FONT_REG, 17 * s)
    f_text_bold = _load_font(FONT_BOLD, 18 * s)
    f_footer = _load_font(FONT_REG, 14 * s)

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

    # --- Barcode reserved area ---
    # DO NOT DRAW BARCODE HERE.
    # The original barcode crop from base.png is pasted as the LAST operation.
    barcode_h = 55 * s
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

    to_fields = [
        (data["recipient_name"], f_text_bold, 26 * s),
        (data["address_line_1"], f_text, 24 * s),
        (data["postal_city"], f_text, 24 * s),
        (data["country"], f_text, 24 * s),
    ]

    if data.get("phone"):
        to_fields.append((data["phone"], f_text, 24 * s))

    total_lines = sum(estimate_lines(text, font, available_w) for text, font, _ in to_fields)

    # Shrink only if address is too long
    if total_lines > 6:
        f_text_small = _load_font(FONT_REG, 14 * s)
        f_text_bold_small = _load_font(FONT_BOLD, 15 * s)
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
                if line:
                    draw.text((tx, ty), line, font=font, fill=TEXT_COLOR)
                    ty += line_h
                line = word
        if line:
            draw.text((tx, ty), line, font=font, fill=TEXT_COLOR)
            ty += line_h
        return ty

    for text, font, line_h in to_fields:
        ty = draw_wrapped(text, font, ty, line_h)

    draw.line(
        [(col_split, y - 5 * s), (col_split, max(fy, ty) + 5 * s)],
        fill=LINE_COLOR_THIN,
        width=2 * s,
    )

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

    # --- Parcel Ref. No QR drawn ---
    draw.text((margin, y), "Parcel Ref:", font=f_text_bold, fill=TEXT_COLOR)
    draw.text((margin, y + 24 * s), "EGT-5400-78910", font=f_text_bold, fill=TEXT_COLOR)

    y += 70 * s
    draw.line([(margin, y), (w - margin, y)], fill=LINE_COLOR_THIN, width=2 * s)
    y += 18 * s

    # --- Footer ---
    draw.text((margin, y), "trackflow.ltd/track", font=f_footer, fill=(95, 95, 100))

    # Downsample with Lanczos for clean text.
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

    # Minimal expansion only to cover old label edges.
    centroid = dest_corners.mean(axis=0)
    expanded_corners = centroid + (dest_corners - centroid) * 1.008

    M = cv2.getPerspectiveTransform(src_corners, expanded_corners.astype(np.float32))
    warped_label = cv2.warpPerspective(label_cv, M, (w, h), flags=cv2.INTER_CUBIC)

    mask = np.ones((lh, lw), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_LINEAR)

    # Light realism on the synthetic label only.
    warped_label = cv2.GaussianBlur(warped_label, (3, 3), 0.3)

    noise = np.random.default_rng(3).normal(0, 1.2, warped_label.shape).astype(np.float32)
    warped_label = np.clip(warped_label.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    sharpen_kernel = np.array([
        [0, -0.04, 0],
        [-0.04, 1.16, -0.04],
        [0, -0.04, 0],
    ])
    warped_label = cv2.filter2D(warped_label, -1, sharpen_kernel)

    warped_mask = cv2.GaussianBlur(warped_mask, (5, 5), 1.5)
    mask_f = (warped_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

    result = base_img_bgr.astype(np.float32) * (1 - mask_f) + warped_label.astype(np.float32) * mask_f
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Whole-image phone-photo realism pass
# ---------------------------------------------------------------------------

def apply_photo_realism(img_bgr: np.ndarray) -> np.ndarray:
    """
    Make the final image look like a slightly compressed phone photo.

    IMPORTANT:
    The locked barcode is pasted AFTER this function, so the barcode itself
    is not blurred, resized, recompressed, or altered.
    """
    h, w = img_bgr.shape[:2]

    # 1. Very light overall blur
    img = cv2.GaussianBlur(img_bgr, (3, 3), 0.5)

    # 2. Slight micro-contrast reduction
    img_f = img.astype(np.float32)
    img_f = img_f * 0.97 + 128 * 0.03
    img = np.clip(img_f, 0, 255).astype(np.uint8)

    # 3. Small downscale/upscale roundtrip
    small = cv2.resize(img, (int(w * 0.92), int(h * 0.92)), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 4. JPEG re-encode to make it look like a real low-quality phone image
    encode_quality = 78
    success, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, encode_quality])
    if success:
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return img


# ---------------------------------------------------------------------------
# Locked barcode paste
# ---------------------------------------------------------------------------

def paste_locked_barcode_last(result_bgr: np.ndarray, base_bgr: np.ndarray, barcode_region=BARCODE_REGION) -> np.ndarray:
    """
    Paste the original barcode crop from base_bgr as the FINAL operation.

    This keeps the barcode visually static in every generated output.
    For true pixel identity, save as PNG and do not run any processing after this.
    """
    x0, y0, x1, y1 = barcode_region

    h, w = result_bgr.shape[:2]
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid BARCODE_REGION: {barcode_region}")

    out = result_bgr.copy()

    # Hard paste: no feather, no blending, no blur.
    # This makes the crop identical to the base image within this rectangle.
    out[y0:y1, x0:x1] = base_bgr[y0:y1, x0:x1]

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_tracking(tracking_number: str) -> str:
    """Normalize to TF-XXXXXXXXXX format."""
    trk = tracking_number or ""
    trk = trk.strip().upper()
    digits_only = "".join(ch for ch in trk if ch.isdigit())

    if trk.startswith("TF-") and len(trk) > 3:
        return trk

    if digits_only:
        return f"TF-{digits_only}"

    return "TF-0000000000"


def generate_label(recipient: dict, tracking_number: str = None, out_path: str = None):
    """
    recipient keys:
        name
        line1
        line2 = city
        line3 = postal code
        line4 = country
        phone optional

    tracking_number:
        "5839174268" -> "TF-5839174268"
        "TF-5839174268" -> "TF-5839174268"

    Returns PIL.Image.Image RGB.
    """
    postal_code = recipient.get("line3", "")
    city = recipient.get("line2", "")

    if postal_code and city:
        postal_city_combined = f"{postal_code} {city}"
    else:
        postal_city_combined = postal_code or city

    data = {
        "recipient_name": recipient["name"],
        "address_line_1": recipient.get("line1", ""),
        "postal_city": postal_city_combined,
        "country": recipient.get("line4", ""),
        "phone": recipient.get("phone", ""),
        "tracking_number": normalize_tracking(tracking_number),
    }

    base_bgr = cv2.imread(BASE_IMAGE_PATH)
    if base_bgr is None:
        raise FileNotFoundError(BASE_IMAGE_PATH)

    paper_color = get_paper_color(base_bgr)
    flat = build_flat_label(data, paper_color)

    result_bgr = warp_and_composite(base_bgr, flat, DEST_CORNERS)

    # QR removed by design: no QR is drawn in the flat label and no QR crop is pasted back.

    # Apply low-quality phone-photo realism BEFORE locking barcode.
    result_bgr = apply_photo_realism(result_bgr)

    # FINAL STEP: lock barcode from base photo.
    # Nothing should run after this except RGB conversion and saving.
    result_bgr = paste_locked_barcode_last(result_bgr, base_bgr, BARCODE_REGION)

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    result_img = Image.fromarray(result_rgb)

    if out_path:
        ext = os.path.splitext(out_path)[1].lower()

        # For actual pixel-identical barcode, PNG is best.
        # JPEG will change pixels during saving, even if the barcode was pasted last.
        if ext in [".jpg", ".jpeg"]:
            result_img.save(out_path, quality=95, subsampling=0)
        else:
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
        out_path="test_v5.png",
    )
    print("done", img.size)

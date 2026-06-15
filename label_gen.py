"""
Deterministic shipping-label compositor.

Builds a flat label image with Pillow (exact, correctly-spelled text),
then warps it onto the base package photo using OpenCV perspective
transform, matching the photographed label's position/angle, with
blur/noise/contrast adjustments for realism.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BASE_IMAGE_PATH = "base.png"

LABEL_W, LABEL_H = 600, 850

# Four corners of the real label in the base photo: TL, TR, BR, BL
DEST_CORNERS = np.array([
    [596, 172],
    [872, 181],
    [935, 494],
    [571, 490],
], dtype=np.float32)

FONT_DIR = "/usr/share/fonts/truetype/liberation"
FONT_BOLD = f"{FONT_DIR}/LiberationSans-Bold.ttf"
FONT_REG = f"{FONT_DIR}/LiberationSans-Regular.ttf"

TEXT_COLOR = (30, 30, 35)
LINE_COLOR = (60, 60, 70)


# ---------------------------------------------------------------------------
# Build the flat label with Pillow
# ---------------------------------------------------------------------------

def build_flat_label(data: dict) -> Image.Image:
    """
    data keys: recipient_name, address_line_1, postal_city, country,
               phone (optional), tracking_number
    """
    label = Image.new("RGB", (LABEL_W, LABEL_H), (200, 217, 238))
    draw = ImageDraw.Draw(label)

    margin = 30
    w = LABEL_W

    f_brand = ImageFont.truetype(FONT_BOLD, 38)
    f_service_small = ImageFont.truetype(FONT_BOLD, 18)
    f_route = ImageFont.truetype(FONT_BOLD, 46)
    f_trk = ImageFont.truetype(FONT_BOLD, 26)
    f_label_small = ImageFont.truetype(FONT_REG, 16)
    f_text = ImageFont.truetype(FONT_REG, 17)
    f_text_bold = ImageFont.truetype(FONT_BOLD, 18)
    f_footer = ImageFont.truetype(FONT_REG, 14)

    y = margin

    # --- Header: TrackFlow / TF-EXPRESS ---
    draw.text((margin, y), "TrackFlow", font=f_brand, fill=TEXT_COLOR)
    tf_text = "TF-EXPRESS"
    tf_w = draw.textlength(tf_text, font=f_service_small)
    draw.text((w - margin - tf_w, y + 14), tf_text, font=f_service_small, fill=TEXT_COLOR)
    y += 60
    draw.line([(margin, y), (w - margin, y)], fill=LINE_COLOR, width=2)
    y += 25

    # --- Route ---
    route_text = "US - DE - DE"
    route_w = draw.textlength(route_text, font=f_route)
    draw.text(((w - route_w) / 2, y), route_text, font=f_route, fill=TEXT_COLOR)
    y += 75
    draw.line([(margin, y), (w - margin, y)], fill=LINE_COLOR, width=2)
    y += 20

    # --- Barcode (decorative) ---
    barcode_h = 55
    rng = np.random.default_rng(7)
    x = margin + 10
    bx_end = w - margin - 10
    while x < bx_end:
        bw = rng.integers(2, 6)
        if rng.random() > 0.4:
            draw.rectangle([x, y, x + bw, y + barcode_h], fill=(20, 20, 20))
        x += bw + rng.integers(2, 5)
    y += barcode_h + 12

    # --- Tracking number ---
    trk_text = f"TRK # {data['tracking_number']}"
    trk_w = draw.textlength(trk_text, font=f_trk)
    draw.text(((w - trk_w) / 2, y), trk_text, font=f_trk, fill=TEXT_COLOR)
    y += 45
    draw.line([(margin, y), (w - margin, y)], fill=LINE_COLOR, width=2)
    y += 18

    # --- From / To columns ---
    col_split = w // 2

    draw.text((margin, y), "From:", font=f_text_bold, fill=TEXT_COLOR)
    fy = y + 24
    for line in [
        "Sunny Global Trading PLLC",
        "200 Pine St, 23rd",
        "Seattle, WA 98101",
        "United States",
        "+1 971-123 4567",
    ]:
        draw.text((margin, fy), line, font=f_text, fill=TEXT_COLOR)
        fy += 24

    draw.line([(col_split, y - 5), (col_split, fy + 5)], fill=LINE_COLOR, width=2)

    tx = col_split + 25
    available_w = w - margin - tx  # width available for To: column text

    # Estimate total wrapped line count to decide font size
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

    f_text_try = f_text
    f_text_bold_try = f_text_bold
    line_h_try = 24
    name_line_h_try = 26

    total_lines = (
        estimate_lines(data["recipient_name"], f_text_bold_try, available_w)
        + estimate_lines(data["address_line_1"], f_text_try, available_w)
        + estimate_lines(data["postal_city"], f_text_try, available_w)
        + estimate_lines(data["country"], f_text_try, available_w)
        + (estimate_lines(data["phone"], f_text_try, available_w) if data.get("phone") else 0)
    )

    if total_lines > 6:
        f_text_try = ImageFont.truetype(FONT_REG, 14)
        f_text_bold_try = ImageFont.truetype(FONT_BOLD, 15)
        line_h_try = 20
        name_line_h_try = 22
        available_w = w - margin - tx  # recompute not needed, same width

    draw.text((tx, y), "To:", font=f_text_bold, fill=TEXT_COLOR)
    ty = y + 24

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

    ty = draw_wrapped(data["recipient_name"], f_text_bold_try, ty, name_line_h_try)
    ty = draw_wrapped(data["address_line_1"], f_text_try, ty, line_h_try)
    ty = draw_wrapped(data["postal_city"], f_text_try, ty, line_h_try)
    ty = draw_wrapped(data["country"], f_text_try, ty, line_h_try)
    if data.get("phone"):
        ty = draw_wrapped(data["phone"], f_text_try, ty, line_h_try)

    y = max(fy, ty) + 15
    draw.line([(margin, y), (w - margin, y)], fill=LINE_COLOR, width=2)
    y += 18

    # --- Service / Weight / Pieces row ---
    col_w = (w - 2 * margin) / 3
    draw.text((margin, y), "Service", font=f_label_small, fill=(90, 90, 100))
    draw.text((margin, y + 22), "EXPRESS", font=f_text_bold, fill=TEXT_COLOR)

    draw.text((margin + col_w, y), "Weight", font=f_label_small, fill=(90, 90, 100))
    draw.text((margin + col_w, y + 22), "12.40 KG", font=f_text_bold, fill=TEXT_COLOR)

    draw.text((margin + 2 * col_w, y), "Pieces", font=f_label_small, fill=(90, 90, 100))
    draw.text((margin + 2 * col_w, y + 22), "1 / 1", font=f_text_bold, fill=TEXT_COLOR)

    y += 60
    draw.line([(margin, y), (w - margin, y)], fill=LINE_COLOR, width=2)
    y += 18

    # --- Parcel Ref + QR placeholder ---
    draw.text((margin, y), "Parcel Ref:", font=f_text_bold, fill=TEXT_COLOR)
    draw.text((margin, y + 24), "EGT-5400-78910", font=f_text_bold, fill=TEXT_COLOR)

    qr_size = 90
    qr_x = w - margin - qr_size
    qr_y = y - 5
    rng2 = np.random.default_rng(11)
    cell = qr_size // 9
    for i in range(9):
        for j in range(9):
            if rng2.random() > 0.5:
                draw.rectangle([
                    qr_x + i * cell, qr_y + j * cell,
                    qr_x + (i + 1) * cell, qr_y + (j + 1) * cell
                ], fill=(20, 20, 20))
    for cx, cy in [(qr_x, qr_y), (qr_x + qr_size - cell * 2, qr_y), (qr_x, qr_y + qr_size - cell * 2)]:
        draw.rectangle([cx, cy, cx + cell * 2, cy + cell * 2], outline=(20, 20, 20), width=3)

    y += 70
    draw.line([(margin, y), (w - margin, y)], fill=LINE_COLOR, width=2)
    y += 18

    # --- Footer ---
    draw.text((margin, y), "trackflow.ltd/track", font=f_footer, fill=(90, 90, 100))

    return label


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

    centroid = dest_corners.mean(axis=0)
    expanded_corners = centroid + (dest_corners - centroid) * 1.04
    M_mask = cv2.getPerspectiveTransform(src_corners, expanded_corners.astype(np.float32))
    warped_label = cv2.warpPerspective(label_cv, M_mask, (w, h))

    mask = np.ones((lh, lw), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask, M_mask, (w, h))

    # Realism adjustments
    warped_label = cv2.GaussianBlur(warped_label, (3, 3), 0.6)
    warped_label = cv2.convertScaleAbs(warped_label, alpha=0.97, beta=2)

    noise = np.random.default_rng(3).normal(0, 4, warped_label.shape).astype(np.float32)
    warped_label = np.clip(warped_label.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    warped_mask = cv2.GaussianBlur(warped_mask, (9, 9), 3)
    mask_f = (warped_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

    result = (base_img_bgr.astype(np.float32) * (1 - mask_f) + warped_label.astype(np.float32) * mask_f)
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_label(recipient: dict, tracking_number: str = None, out_path: str = None):
    """
    recipient: dict with keys name, line1 (street), line2 (city), line3
               (postal code), line4 (country), phone
    tracking_number: string, will be displayed as 'TRK # <tracking_number>'
    Returns a PIL.Image.Image (RGB). If out_path is given, also saves to disk.
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
        "tracking_number": tracking_number or "",
    }

    flat = build_flat_label(data)

    base_bgr = cv2.imread(BASE_IMAGE_PATH)
    if base_bgr is None:
        raise FileNotFoundError(BASE_IMAGE_PATH)

    result_bgr = warp_and_composite(base_bgr, flat, DEST_CORNERS)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    result_img = Image.fromarray(result_rgb)

    if out_path:
        result_img.save(out_path)
    return result_img


if __name__ == "__main__":
    img = generate_label(
        {
            "name": "Skylar Lippert",
            "line1": "Lise-Meitner-Strasse 21",
            "line2": "Geilenkirchen",
            "line3": "52511",
            "line4": "Germany",
            "phone": "+49 301 123 4567",
        },
        tracking_number="2550 1367 9724 782",
        out_path="test_output.png",
    )
    print("done", img.size)

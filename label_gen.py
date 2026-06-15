"""
Shipping-label patcher v5.

Preserves the original photographed label almost entirely. Only three
regions are modified:
  1. Tracking number line -> new "TF-XXXXXXXXXX" text
  2. "To:" recipient block -> new name/address
  3. QR code area -> removed (patched with paper texture)

Each region is covered with a realistic paper-texture patch (sampled from
clean nearby areas of the real photo) and new text is rendered at high
resolution then downsampled for crisp, minimally-blurred results.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import textwrap

BASE_IMAGE_PATH = "base.png"

FONT_DIR = "/usr/share/fonts/truetype/liberation"
FONT_BOLD = f"{FONT_DIR}/LiberationSans-Bold.ttf"
FONT_REG = f"{FONT_DIR}/LiberationSans-Regular.ttf"

TEXT_COLOR = (25, 25, 30)

SCALE = 4  # supersampling factor for crisp text rendering

# Regions in original base.png coordinates: (x0, y0, x1, y1)
TRK_REGION = (615, 258, 860, 282)
TO_REGION = (750, 286, 893, 363)
QR_REGION = (820, 408, 912, 478)

# Slight perspective tilt of the label (degrees) - used to rotate rendered
# text patches so they match the label's angle
ANGLE = -2.0


# ---------------------------------------------------------------------------
# Texture patch helpers
# ---------------------------------------------------------------------------

PAPER_COLOR = (198, 216, 240)  # median paper tone sampled from base photo


def make_texture_patch(shape_hw, seed=0):
    """
    Build a realistic paper-texture patch of size (h, w) using the sampled
    paper color plus subtle correlated noise (grain), avoiding flat/sticker
    look while not requiring a donor region.
    """
    h, w = shape_hw
    base = np.array(PAPER_COLOR, dtype=np.float32)
    patch = np.tile(base, (h, w, 1))

    rng = np.random.default_rng(seed)
    luminance_noise = rng.normal(0, 1, size=(h, w, 1)) * 3.5
    patch = patch + np.repeat(luminance_noise, 3, axis=2)
    patch = np.clip(patch, 0, 255).astype(np.uint8)
    patch = cv2.GaussianBlur(patch, (3, 3), 0.4)
    return patch


def paste_with_feather(base_rgb: np.ndarray, patch: np.ndarray, region, feather=4):
    """Paste `patch` into base_rgb at `region` with feathered edges."""
    x0, y0, x1, y1 = region
    ph, pw = patch.shape[:2]

    mask = np.ones((ph, pw), dtype=np.float32)
    for i in range(feather):
        a = (i + 1) / feather
        mask[i, :] *= a
        mask[-(i + 1), :] *= a
        mask[:, i] *= a
        mask[:, -(i + 1)] *= a
    mask = cv2.GaussianBlur(mask, (5, 5), 1.5)
    mask3 = mask[:, :, np.newaxis]

    region_pixels = base_rgb[y0:y1, x0:x1].astype(np.float32)
    blended = region_pixels * (1 - mask3) + patch.astype(np.float32) * mask3
    base_rgb[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# High-res text rendering -> rotated -> composited into a region
# ---------------------------------------------------------------------------

def render_text_block(lines, region_w, region_h, angle=ANGLE):
    """
    Render `lines` (list of (text, font_size_pt, bold, line_height_pt))
    onto a high-res transparent canvas sized for `region_w` x `region_h`
    (in base-photo pixels), then rotate by `angle` and downsample.
    Returns an RGBA PIL Image the same size as (region_w, region_h)
    (slightly larger to allow rotation without clipping, caller should
    crop/paste accordingly using the returned offset).
    """
    s = SCALE
    canvas_w, canvas_h = region_w * s, region_h * s
    # Extra padding so rotation doesn't clip content
    pad = int(0.15 * max(canvas_w, canvas_h))
    layer = Image.new("RGBA", (canvas_w + 2 * pad, canvas_h + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    y = pad + 2 * s
    for text, font_size, bold, line_h in lines:
        font_path = FONT_BOLD if bold else FONT_REG
        font = ImageFont.truetype(font_path, font_size * s)
        draw.text((pad + 2 * s, y), text, font=font, fill=(*TEXT_COLOR, 255))
        y += line_h * s

    rotated = layer.rotate(angle, resample=Image.BICUBIC, expand=False)

    # Downsample
    final = rotated.resize(
        (rotated.width // s, rotated.height // s), Image.LANCZOS
    )
    return final, pad // s


def composite_text(base_rgb: np.ndarray, region, lines, angle=ANGLE):
    x0, y0, x1, y1 = region
    rw, rh = x1 - x0, y1 - y0

    text_img, pad = render_text_block(lines, rw, rh, angle)

    # text_img is (rw + 2*pad) x (rh + 2*pad); paste centered over region,
    # extending pad pixels beyond region bounds (clip at image edges)
    tw, th = text_img.size
    px0 = x0 - pad
    py0 = y0 - pad

    # Composite RGBA text_img onto base_rgb at (px0, py0), clipping bounds
    H, W = base_rgb.shape[:2]
    sx0, sy0 = max(0, -px0), max(0, -py0)
    dx0, dy0 = max(0, px0), max(0, py0)
    ex = min(W, px0 + tw)
    ey = min(H, py0 + th)
    sx1 = sx0 + (ex - dx0)
    sy1 = sy0 + (ey - dy0)

    if ex <= dx0 or ey <= dy0:
        return

    text_arr = np.array(text_img)
    alpha = text_arr[sy0:sy1, sx0:sx1, 3:4].astype(np.float32) / 255.0
    rgb = text_arr[sy0:sy1, sx0:sx1, :3].astype(np.float32)

    target = base_rgb[dy0:ey, dx0:ex].astype(np.float32)
    blended = target * (1 - alpha) + rgb * alpha
    base_rgb[dy0:ey, dx0:ex] = np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_label(recipient: dict, tracking_number: str = None, out_path: str = None):
    """
    recipient: dict with keys name, line1 (street), line2 (city), line3
               (postal code), line4 (country), phone (optional)
    tracking_number: digits (or 'TF-XXXXXXXXXX'); will be normalized to
                     'TF-<digits>'.
    Returns a PIL.Image.Image (RGB). If out_path is given, also saves to disk.
    """
    base_bgr = cv2.imread(BASE_IMAGE_PATH)
    if base_bgr is None:
        raise FileNotFoundError(BASE_IMAGE_PATH)
    base_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)

    # --- Normalize tracking number to TF-<digits> ---
    trk = tracking_number or ""
    digits_only = "".join(ch for ch in trk if ch.isdigit())
    if not digits_only:
        # Derive a stable numeric fallback from any provided string
        digits_only = "".join(str(ord(c) % 10) for c in trk)[:10] or "0000000000"
    trk_formatted = f"TF-{digits_only}"

    # --- 1. Patch & redraw tracking number ---
    trk_h = TRK_REGION[3] - TRK_REGION[1]
    trk_w = TRK_REGION[2] - TRK_REGION[0]
    trk_patch = make_texture_patch((trk_h, trk_w), seed=11)
    paste_with_feather(base_rgb, trk_patch, TRK_REGION, feather=4)

    composite_text(
        base_rgb, TRK_REGION,
        lines=[(trk_formatted, 19, True, trk_h)],
        angle=ANGLE,
    )

    # --- 2. Patch & redraw To: block ---
    to_w = TO_REGION[2] - TO_REGION[0]
    to_h = TO_REGION[3] - TO_REGION[1]
    to_patch = make_texture_patch((to_h, to_w), seed=12)
    paste_with_feather(base_rgb, to_patch, TO_REGION, feather=4)

    # Build To: text lines
    postal_code = recipient.get("line3", "")
    city = recipient.get("line2", "")
    if postal_code and city:
        postal_city = f"{postal_code} {city}"
    else:
        postal_city = postal_code or city

    fields = [
        ("To:", 13, False),
        (recipient["name"], 14, True),
        (recipient.get("line1", ""), 13, False),
        (postal_city, 13, False),
        (recipient.get("line4", ""), 13, False),
    ]
    if recipient.get("phone"):
        fields.append((recipient["phone"], 13, False))

    # Wrap long fields
    max_chars = 24
    max_chars_name = 18
    wrapped_lines = []
    for text, size, bold in fields:
        if not text:
            continue
        width_limit = max_chars_name if bold else max_chars
        for w in textwrap.wrap(text, width=width_limit) or [""]:
            wrapped_lines.append((w, size, bold))

    n_lines = len(wrapped_lines)
    line_h = 11 if n_lines <= 7 else max(8, int(11 * 7 / n_lines))
    if n_lines > 7:
        wrapped_lines = [(t, max(9, sz - 2), b) for t, sz, b in wrapped_lines]

    text_lines = [(t, sz, b, line_h + (2 if b and t != "To:" else 0)) for t, sz, b in wrapped_lines]

    composite_text(base_rgb, TO_REGION, lines=text_lines, angle=ANGLE)

    # --- 3. Remove QR: patch with paper texture ---
    qr_w = QR_REGION[2] - QR_REGION[0]
    qr_h = QR_REGION[3] - QR_REGION[1]
    qr_patch = make_texture_patch((qr_h, qr_w), seed=13)
    paste_with_feather(base_rgb, qr_patch, QR_REGION, feather=3)

    result_img = Image.fromarray(base_rgb)
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
        out_path="test_v5.png",
    )
    print("done", img.size)

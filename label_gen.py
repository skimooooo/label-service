from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import textwrap

BASE_IMG = "base.png"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
FONT_REG_PATH = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
TEXT_COLOR = (35, 40, 50, 255)
ANGLE = -2.0


def make_texture_patch(base, region, top_strip_y, bot_strip_y, seed=42):
    """
    Build a realistic paper-texture patch for `region` (x0,y0,x1,y1) by
    vertically blending real pixel strips sampled from top_strip_y and
    bot_strip_y (each a (y0,y1) tuple of clean blank rows), with noise
    derived from those strips' own variance.
    """
    x0, y0, x1, y1 = region
    pw, ph = x1 - x0, y1 - y0

    top_strip = base.crop((x0, top_strip_y[0], x1, top_strip_y[1]))
    bot_strip = base.crop((x0, bot_strip_y[0], x1, bot_strip_y[1]))

    top_arr = np.array(top_strip).astype(float)
    bot_arr = np.array(bot_strip).astype(float)

    top_row = top_arr.mean(axis=0)
    bot_row = bot_arr.mean(axis=0)

    patch = np.zeros((ph, pw, 3), dtype=float)
    for y in range(ph):
        t = y / max(ph - 1, 1)
        patch[y] = top_row * (1 - t) + bot_row * t

    noise_source = np.concatenate([top_arr - top_row, bot_arr - bot_row], axis=0)
    noise_std = noise_source.std(axis=(0, 2)).mean()
    rng = np.random.default_rng(seed)
    luminance_noise = rng.normal(0, 1, size=(ph, pw, 1)) * noise_std * 0.5
    noise = np.repeat(luminance_noise, 3, axis=2)
    patch = np.clip(patch + noise, 0, 255).astype('uint8')

    patch_img = Image.fromarray(patch, 'RGB')
    return patch_img.filter(ImageFilter.GaussianBlur(0.4))


def paste_with_feather(base_img, patch_img, position, feather=6):
    """
    Paste patch_img onto base_img at position with a feathered (soft-edge)
    alpha mask so the patch blends gradually into surrounding pixels.
    """
    pw, ph = patch_img.size
    # Build alpha mask: fully opaque in center, fading to 0 over `feather` px at edges
    mask = Image.new("L", (pw, ph), 255)
    mask_draw = ImageDraw.Draw(mask)
    for i in range(feather):
        alpha = int(255 * (i + 1) / feather)
        mask_draw.rectangle([i, i, pw - 1 - i, ph - 1 - i], outline=alpha)
    mask = mask.filter(ImageFilter.GaussianBlur(feather / 2))

    base_img.paste(patch_img, position, mask)


def generate_label(recipient: dict, tracking_number: str = None, out_path: str = None):
    base = Image.open(BASE_IMG).convert("RGB")
    patched = base.copy()

    # --- Address block ---
    # Region: (x0,y0,x1,y1), donor strips above/below
    addr_region = (753, 291, 890, 363)
    addr_top_strip = (363, 366)
    addr_bot_strip = (363, 366)

    addr_patch = make_texture_patch(base, addr_region, addr_top_strip, addr_bot_strip, seed=42)
    paste_with_feather(patched, addr_patch, (addr_region[0], addr_region[1]), feather=5)

    # Build recipient text lines
    lines = []

    def wrap_field(text, max_chars):
        return textwrap.wrap(text, width=max_chars) or [""]

    MAX_CHARS_NAME = 18
    MAX_CHARS = 24

    for w in wrap_field(recipient["name"], MAX_CHARS_NAME):
        lines.append((w, True))
    for key in ("line1", "line2", "line3", "line4"):
        val = recipient.get(key, "")
        if val:
            for w in wrap_field(val, MAX_CHARS):
                lines.append((w, False))
    if recipient.get("phone"):
        lines.append((recipient["phone"], False))

    n_lines = len(lines)
    base_font_size_bold = 14
    base_font_size_reg = 12
    base_line_height = 10
    max_lines_default = 6

    if n_lines > max_lines_default:
        shrink = max_lines_default / n_lines
        font_size_bold = max(9, int(base_font_size_bold * shrink))
        font_size_reg = max(8, int(base_font_size_reg * shrink))
        line_height = max(7, int(base_line_height * shrink))
    else:
        font_size_bold = base_font_size_bold
        font_size_reg = base_font_size_reg
        line_height = base_line_height

    font_bold = ImageFont.truetype(FONT_BOLD_PATH, font_size_bold)
    font_reg = ImageFont.truetype(FONT_REG_PATH, font_size_reg)

    pw, ph = addr_region[2] - addr_region[0], addr_region[3] - addr_region[1]
    layer_w, layer_h = pw + 30, ph + 30
    text_layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)

    font_label = ImageFont.truetype(FONT_REG_PATH, 13)

    # "To:" header
    y = 2
    draw.text((2, y), "To:", font=font_label, fill=TEXT_COLOR)
    y += 16

    for text, is_bold in lines:
        font = font_bold if is_bold else font_reg
        draw.text((2, y), text, font=font, fill=TEXT_COLOR)
        y += line_height + (2 if is_bold else 0)

    rotated = text_layer.rotate(ANGLE, resample=Image.BICUBIC, expand=False)
    patched.paste(rotated, (addr_region[0] - 2, addr_region[1] - 2), rotated)

    # --- Tracking number ---
    if tracking_number:
        trk_region = (623, 263, 855, 282)
        trk_top_strip = (263, 267)
        trk_bot_strip = (283, 287)

        trk_patch = make_texture_patch(base, trk_region, trk_top_strip, trk_bot_strip, seed=43)
        paste_with_feather(patched, trk_patch, (trk_region[0], trk_region[1]), feather=5)

        tpw, tph = trk_region[2] - trk_region[0], trk_region[3] - trk_region[1]
        trk_layer = Image.new("RGBA", (tpw + 20, tph + 10), (0, 0, 0, 0))
        trk_draw = ImageDraw.Draw(trk_layer)
        trk_font = ImageFont.truetype(FONT_BOLD_PATH, 17)
        trk_draw.text((5, 1), f"TRK # {tracking_number}", font=trk_font, fill=(20, 20, 20, 255))
        trk_rotated = trk_layer.rotate(ANGLE, resample=Image.BICUBIC, expand=False)
        patched.paste(trk_rotated, (trk_region[0] - 2, trk_region[1] - 1), trk_rotated)

    if out_path:
        patched.save(out_path)
    return patched


if __name__ == "__main__":
    # Test 1: short German address
    img1 = generate_label(
        {
            "name": "Sarah Klein",
            "line1": "Hauptstr. 45",
            "line2": "Munich",
            "line3": "80331",
            "line4": "Germany",
            "phone": "+49 170 234 5678",
        },
        tracking_number="9981 4523 7710",
        out_path="final_v3_short.png",
    )

    # Test 2: long Libyan address
    img2 = generate_label(
        {
            "name": "Mohamed Al-Bargathi",
            "line1": "Souk Al Thalatha, Hay Al Andalus",
            "line2": "Tripoli",
            "line3": "",
            "line4": "Libya",
            "phone": "+218 92 555 1234",
        },
        tracking_number="ARE-2026-44120",
        out_path="final_v3_long.png",
    )

    print("done")

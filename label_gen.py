from PIL import Image, ImageDraw, ImageFont, ImageFilter
import textwrap

BASE_IMG = "base.png"

# Address block bbox in original image coords
X0, Y0, X1, Y1 = 750, 287, 855, 355
ANGLE = -2.0

FONT_BOLD_PATH = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
FONT_REG_PATH = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"

TEXT_COLOR = (35, 40, 50, 255)


def generate_label(recipient: dict, tracking_number: str = None, out_path: str = None):
    """
    recipient: dict with keys: name, line1 (street), line2 (city), line3 (postal code),
               line4 (country), phone
    tracking_number: optional, replaces the TRK# on the label
    """
    base = Image.open(BASE_IMG).convert("RGB")
    # Address block bbox in original image coords (widened slightly to avoid overflow)
    x0, y0, x1, y1 = X0, Y0, X1 + 15, Y1
    patch_w, patch_h = x1 - x0, y1 - y0

    # Patch out old text with sampled background
    sample_strip = base.crop((x1 + 2, y0, x1 + 10, y1))
    sample_strip = sample_strip.resize((patch_w, patch_h), Image.BILINEAR)
    sample_strip = sample_strip.filter(ImageFilter.GaussianBlur(0.8))

    patched = base.copy()
    patched.paste(sample_strip, (x0, y0))

    # Build lines, wrapping long fields
    lines = []  # list of (text, is_bold)

    def wrap_field(text, max_chars):
        return textwrap.wrap(text, width=max_chars) or [""]

    # Max characters per line before wrapping (tuned for ~13-15px font, ~120px width)
    MAX_CHARS_NAME = 18   # bold font is wider per character
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

    # Auto-shrink font if too many lines (base design assumes 6 lines)
    base_font_size_bold = 15
    base_font_size_reg = 13
    base_line_height = 11
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

    layer_w, layer_h = patch_w + 30, patch_h + 30
    text_layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)

    y = 2
    for text, is_bold in lines:
        font = font_bold if is_bold else font_reg
        draw.text((2, y), text, font=font, fill=TEXT_COLOR)
        y += line_height + (2 if is_bold else 0)

    rotated = text_layer.rotate(ANGLE, resample=Image.BICUBIC, expand=False)
    patched.paste(rotated, (x0 - 2, y0 - 2), rotated)

    if tracking_number:
        patched = replace_tracking_number(patched, tracking_number)

    if out_path:
        patched.save(out_path)
    return patched


def replace_tracking_number(img: Image.Image, tracking_number: str):
    # TRK # line bbox in original image coords
    tx0, ty0, tx1, ty1 = 623, 263, 855, 282
    pw, ph = tx1 - tx0, ty1 - ty0

    # Synthetic paper-color patch (sampled clean label color, bluish-white)
    base_color = (199, 217, 241)
    patch = Image.new("RGB", (pw, ph), base_color)
    # add a very subtle gradient to mimic lighting falloff toward the right edge
    import numpy as np
    arr = np.array(patch).astype(float)
    grad = np.linspace(0, -10, pw)  # slightly darker toward right
    arr[:, :, 0] += grad[np.newaxis, :]
    arr[:, :, 1] += grad[np.newaxis, :]
    arr[:, :, 2] += grad[np.newaxis, :]
    arr = np.clip(arr, 0, 255).astype('uint8')
    patch = Image.fromarray(arr)
    patch = patch.filter(ImageFilter.GaussianBlur(0.3))

    img = img.copy()
    img.paste(patch, (tx0, ty0))

    layer = Image.new("RGBA", (pw + 20, ph + 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = ImageFont.truetype(FONT_BOLD_PATH, 17)
    draw.text((5, 1), f"TRK # {tracking_number}", font=font, fill=(20, 20, 20, 255))
    rotated = layer.rotate(ANGLE, resample=Image.BICUBIC, expand=False)
    img.paste(rotated, (tx0 - 2, ty0 - 1), rotated)
    return img


if __name__ == "__main__":
    # Test case 1: short German address
    generate_label(
        {
            "name": "Sarah Klein",
            "line1": "Hauptstr. 45",
            "line2": "Munich",
            "line3": "80331",
            "line4": "Germany",
            "phone": "+49 170 234 5678",
        },
        tracking_number="9981 4523 7710",
        out_path="test_short.png",
    )

    # Test case 2: long name + long street (Libyan style)
    generate_label(
        {
            "name": "Abdulrahman Mohammed Al-Zawawi",
            "line1": "Souk Al Jumaa Street, Building 12, Near Al Noor Pharmacy",
            "line2": "Tripoli",
            "line3": "",
            "line4": "Libya",
            "phone": "+218 91 234 5678",
        },
        tracking_number="LY-2026-008812",
        out_path="test_long.png",
    )

    print("done")

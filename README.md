# Shipping Label Image Generator

Generates a realistic shipping-label-on-package image with a custom
recipient address and tracking number composited onto a base photo.
Cost: $0 per image (no AI image generation), <1 second per request.

## Files
- `base.png` — the base package photo (label area gets overwritten per request)
- `label_gen.py` — core compositing logic (PIL/Pillow)
- `main.py` — FastAPI wrapper exposing `/generate-label`
- `requirements.txt` — Python deps

## Run locally
```
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API

POST `/generate-label`

```json
{
  "recipient": {
    "name": "Mohamed Al-Bargathi",
    "line1": "Souk Al Thalatha, Hay Al Andalus",
    "line2": "Tripoli",
    "line3": "",
    "line4": "Libya",
    "phone": "+218 92 555 1234"
  },
  "tracking_number": "ARE-2026-44120",
  "response_format": "url"
}
```

`response_format`:
- `"url"` (default) → returns raw PNG bytes (`image/png`), suitable for
  direct upload to Supabase storage from your backend
- `"base64"` → returns `{"image_base64": "...", "mime_type": "image/png"}`

## Deploying (free options)
- **Render** (free web service): push this folder as a repo, set start
  command to `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Railway**: same, auto-detects Python + uvicorn
- Both give you a public HTTPS URL to call from your Lovable/Supabase
  edge function.

## Wiring into your funnel
1. Order placed → confirmation email (existing flow)
2. Scheduled job (Supabase `pg_cron` or edge function) finds orders
   ~2 days old without a "tracking email sent" flag
3. For each, call this service with the customer's address + tracking
   number from your order data
4. Upload the returned image to Supabase storage → get public URL
5. Send "your package is on its way" email with that image embedded
6. Mark order as tracking-email-sent

## Customizing the base label
The pixel coordinates for the address block and tracking-number patch
are defined as constants near the top of `label_gen.py`
(`X0, Y0, X1, Y1` and inside `replace_tracking_number`). If you swap in
a different base photo, you'll need to re-measure these coordinates
to match the new label's position/rotation.

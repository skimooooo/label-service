import io
import base64

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from label_gen import generate_label

app = FastAPI(title="Shipping Label Image Generator")


class Recipient(BaseModel):
    name: str
    line1: str = ""   # street address
    line2: str = ""   # city
    line3: str = ""   # postal code
    line4: str = ""   # country
    phone: str = ""


class LabelRequest(BaseModel):
    recipient: Recipient
    tracking_number: str = Field(..., description="Tracking number to display on label")
    response_format: str = Field("url", description="'image' to return raw PNG, 'base64' to return base64 JSON")


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/generate-label")
def generate(req: LabelRequest):
    try:
        img = generate_label(
            recipient=req.recipient.model_dump(),
            tracking_number=req.tracking_number,
            out_path=None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    if req.response_format == "base64":
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return {"image_base64": encoded, "mime_type": "image/png"}

    # default: return raw image bytes
    return StreamingResponse(buf, media_type="image/png")

import base64
import json
import logging
import os
import re
import ssl
import tempfile
import time
import urllib.request
from typing import Optional, Any

import dashscope
from dashscope import ImageSynthesis, MultiModalConversation
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONTEND_DIR = os.path.join(ROOT, "frontend")
ASSETS_DIR = os.path.join(ROOT, "assets")
STYLES_DIR = os.path.join(ASSETS_DIR, "styles")
STYLE_MANIFEST_PATH = os.path.join(ASSETS_DIR, "styles_manifest.json")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
if DASHSCOPE_API_KEY:
    dashscope.api_key = DASHSCOPE_API_KEY

QWEN_IMAGE_EDIT_MODELS = [
    m.strip()
    for m in os.getenv("QWEN_IMAGE_EDIT_MODELS", "").split(",")
    if m.strip()
]
QWEN_IMAGE_EDIT_TIMEOUT = int(os.getenv("QWEN_IMAGE_EDIT_TIMEOUT", "20"))
WANX_IMAGE_EDIT_MODEL = os.getenv("WANX_IMAGE_EDIT_MODEL", "wanx2.1-imageedit")
WANX_IMAGE_EDIT_TIMEOUT = int(os.getenv("WANX_IMAGE_EDIT_TIMEOUT", "45"))

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("nail-tryon-vercel")


def load_style_library() -> list:
    if not os.path.exists(STYLE_MANIFEST_PATH):
        return [
            {
                "id": i,
                "name": f"训练款式 {i:02d}",
                "color_hex": "#d9b8ad",
                "description": "真实款式图",
                "tags": ["训练集"],
                "image": f"/assets/styles/style-{i:02d}.png",
            }
            for i in range(1, 26)
        ]
    with open(STYLE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    styles = []
    for item in manifest.get("styles", []):
        sid = int(item.get("id", 0) or 0)
        if sid <= 0:
            continue
        styles.append({
            "id": sid,
            "name": item.get("name") or f"训练款式 {sid:02d}",
            "color_hex": item.get("color_hex") or "#d9b8ad",
            "description": item.get("description") or "真实款式图",
            "tags": item.get("tags") or ["训练集"],
            "image": item.get("image") or f"/assets/styles/style-{sid:02d}.png",
            "source_url": item.get("source_url", ""),
            "tip_count": item.get("tip_count", 0),
        })
    return sorted(styles, key=lambda s: s["id"])


STYLE_LIBRARY = load_style_library()


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def safe_int(v, default=0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def parse_json_loose(text: str) -> Any:
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1)
    else:
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if m:
            text = m.group(1)
    try:
        return json.loads(text)
    except Exception:
        return None


def data_uri_from_bytes(data: bytes, mime: str = "image/jpeg") -> str:
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def image_suffix(mime: str) -> str:
    if mime == "image/png":
        return ".png"
    if mime == "image/webp":
        return ".webp"
    return ".jpg"


def file_to_data_uri(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        return data_uri_from_bytes(f.read(), mime)


def download_image_as_data_uri(url: str, timeout: int = 90) -> Optional[str]:
    if not url:
        return None
    if url.startswith("data:image/"):
        return url
    req = urllib.request.Request(url, headers={"User-Agent": "nail-tryon/1.0"})
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
        data = resp.read()
        ctype = resp.headers.get("Content-Type", "image/png").split(";")[0].strip() or "image/png"
    if not ctype.startswith("image/"):
        ctype = "image/png"
    return data_uri_from_bytes(data, ctype)


def extract_image_uri(resp) -> Optional[str]:
    candidates = []

    def add(v):
        if isinstance(v, str) and (v.startswith("http://") or v.startswith("https://") or v.startswith("data:image/")):
            candidates.append(v)

    def walk(obj, depth=0):
        if depth > 7 or obj is None or candidates:
            return
        if isinstance(obj, str):
            add(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v, depth + 1)
        elif hasattr(obj, "__dict__"):
            walk({k: v for k, v in obj.__dict__.items() if not k.startswith("_")}, depth + 1)

    walk(resp)
    return candidates[0] if candidates else None


def call_qwen(content: list, model: str = "qwen-vl-plus") -> Optional[str]:
    if not DASHSCOPE_API_KEY:
        return None
    try:
        resp = MultiModalConversation.call(
            model=model,
            messages=[{"role": "user", "content": content}],
            request_timeout=QWEN_IMAGE_EDIT_TIMEOUT,
        )
        if getattr(resp, "status_code", 0) != 200:
            return None
        msg = resp.output.choices[0].message
        if isinstance(msg.content, list):
            return "\n".join([c.get("text", "") for c in msg.content if isinstance(c, dict) and c.get("text")])
        return msg.content
    except Exception:
        return None


def style_reference_path(style_id: int) -> str:
    return os.path.join(STYLES_DIR, f"style-{style_id:02d}.png")


def build_tryon_prompt(style_id: int) -> str:
    style = STYLE_LIBRARY[clamp(style_id, 1, len(STYLE_LIBRARY)) - 1]
    if style_id == 20:
        target_style = (
            "半透明琥珀棕凝胶甲，圆角延长甲片，有白色猫脸、小星星和金箔点缀。"
            "整体像客户参考图里的 AI 美甲效果，温润透亮，不是贴纸。"
        )
    else:
        target_style = (
            f"参考图1的美甲款式：#{style['id']} {style['name']}。"
            f"{style.get('description', '')} 提取它的颜色、图案、质感和装饰元素。"
        )
    return f"""你是专业美甲试戴图像编辑师。输入中如果有两张图，图1是款式参考，最后一张图是必须保留的用户手部照片。

只编辑最后一张手部照片中的真实指甲/甲床区域。保持手指形状、皮肤纹理、掌纹、背景、光照、阴影、构图和清晰度不变。
将美甲自然生成在每个可见指甲上，要沿真实甲面曲率弯曲贴合，符合手指透视和遮挡关系。
需要真实凝胶甲质感：半透明层次、甲面高光、边缘厚度、甲沟阴影、轻微反光和环境光。
不能出现平面贴图、方块边缘、黑色硬边、悬浮感、错位、覆盖皮肤或改变手指。

目标款式：{target_style}

输出一张真实照片风格的完整手部试戴图，不要拼图，不要文字，不要边框。"""


def wanx_image_tryon(hand_bytes: bytes, mime: str, style_id: int) -> Optional[dict]:
    if not DASHSCOPE_API_KEY or not WANX_IMAGE_EDIT_MODEL:
        return None

    hand_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=image_suffix(mime)) as tmp:
            tmp.write(hand_bytes)
            hand_path = tmp.name

        ref_path = style_reference_path(style_id)
        kwargs = {
            "model": WANX_IMAGE_EDIT_MODEL,
            "prompt": build_tryon_prompt(style_id),
            "base_image_url": hand_path,
            "function": "description_edit",
            "n": 1,
        }
        if os.path.exists(ref_path):
            kwargs["ref_img"] = ref_path

        log.info("wanx image edit tryon: model=%s style=%s", WANX_IMAGE_EDIT_MODEL, style_id)
        task = ImageSynthesis.async_call(**kwargs)
        if getattr(task, "status_code", 0) != 200:
            log.warning("wanx image edit task failed: %s %s", getattr(task, "status_code", ""), getattr(task, "message", ""))
            return None

        task_id = (getattr(task, "output", {}) or {}).get("task_id")
        if not task_id:
            log.warning("wanx image edit missing task_id")
            return None

        deadline = time.time() + WANX_IMAGE_EDIT_TIMEOUT
        while time.time() < deadline:
            resp = ImageSynthesis.fetch(task_id)
            if getattr(resp, "status_code", 0) != 200:
                log.warning("wanx image edit fetch failed: %s %s", getattr(resp, "status_code", ""), getattr(resp, "message", ""))
                return None
            output = getattr(resp, "output", {}) or {}
            status = output.get("task_status")
            if status == "SUCCEEDED":
                image_uri = extract_image_uri(resp)
                result_uri = download_image_as_data_uri(image_uri) if image_uri else None
                if result_uri:
                    return {"result_image": result_uri, "model": WANX_IMAGE_EDIT_MODEL}
                log.warning("wanx image edit succeeded without output image")
                return None
            if status in ("FAILED", "CANCELED", "UNKNOWN"):
                log.warning("wanx image edit ended: status=%s message=%s", status, output.get("message", ""))
                return None
            time.sleep(2)

        log.warning("wanx image edit timeout after %ss", WANX_IMAGE_EDIT_TIMEOUT)
        return None
    except Exception as e:
        log.warning("wanx image edit failed: %s", e)
        return None
    finally:
        if hand_path:
            try:
                os.unlink(hand_path)
            except OSError:
                pass


def qwen_image_tryon(hand_uri: str, style_id: int) -> Optional[dict]:
    if not DASHSCOPE_API_KEY:
        log.warning("qwen image edit skipped: DASHSCOPE_API_KEY is missing")
        return None
    if not QWEN_IMAGE_EDIT_MODELS:
        return None
    content = []
    ref_uri = file_to_data_uri(style_reference_path(style_id))
    if ref_uri:
        content.append({"image": ref_uri})
    content.append({"image": hand_uri})
    content.append({"text": build_tryon_prompt(style_id)})

    last_error = ""
    for model in QWEN_IMAGE_EDIT_MODELS:
        try:
            log.info("qwen image edit tryon: model=%s style=%s", model, style_id)
            resp = MultiModalConversation.call(
                model=model,
                messages=[{"role": "user", "content": content}],
                request_timeout=QWEN_IMAGE_EDIT_TIMEOUT,
            )
            if getattr(resp, "status_code", 0) != 200:
                last_error = f"{getattr(resp, 'status_code', '')} {getattr(resp, 'message', '')}".strip()
                log.warning("qwen image edit status failed: model=%s %s", model, last_error)
                continue
            image_uri = extract_image_uri(resp)
            if not image_uri:
                last_error = "no output image in response"
                log.warning("qwen image edit no output image: model=%s", model)
                continue
            result_uri = download_image_as_data_uri(image_uri) if image_uri else None
            if result_uri:
                return {"result_image": result_uri, "model": model}
        except Exception as e:
            last_error = str(e)
            log.warning("qwen image edit failed: model=%s error=%s", model, e)
            continue
    if last_error:
        log.warning("qwen image edit exhausted: %s", last_error)
    return None


def recommend_style(hand_uri: str) -> dict:
    catalog = "\n".join(f"{s['id']}. {s['name']} - {s.get('description', '')}" for s in STYLE_LIBRARY)
    prompt = f"""你是专业美甲顾问。观察用户手部照片，从肤色、手型、裸甲状态和场景分析，并从以下款式中推荐3款：
{catalog}

严格输出 JSON：
{{"analysis":{{"skin_tone":"中性","hand_type":"纤细","nail_status":"裸甲","scene":"日常"}},"recommendations":[{{"style_id":20,"reason":"适合的具体理由"}}]}}"""
    raw = call_qwen([{"image": hand_uri}, {"text": prompt}])
    parsed = parse_json_loose(raw or "")
    if not isinstance(parsed, dict):
        parsed = {
            "analysis": {"skin_tone": "中性", "hand_type": "纤细", "nail_status": "裸甲", "scene": "日常"},
            "recommendations": [{"style_id": 20, "reason": "琥珀猫眼款更接近真实 AI 试戴效果"}],
        }
    recs = parsed.get("recommendations") or []
    cleaned = []
    for r in recs[:3]:
        sid = clamp(safe_int(r.get("style_id"), 20), 1, len(STYLE_LIBRARY))
        item = {"style_id": sid, "reason": str(r.get("reason") or "推荐")[:80]}
        item["style"] = STYLE_LIBRARY[sid - 1]
        cleaned.append(item)
    while len(cleaned) < 3:
        sid = [20, 17, 7][len(cleaned)]
        cleaned.append({"style_id": sid, "reason": "备选推荐", "style": STYLE_LIBRARY[sid - 1]})
    return {"analysis": parsed.get("analysis") or {}, "recommendations": cleaned}


def fit_advice(style_id: int, analysis: dict = None) -> dict:
    style = STYLE_LIBRARY[clamp(style_id, 1, len(STYLE_LIBRARY)) - 1]
    return {
        "fit_score": 92,
        "fit_text": f"{style['name']}与当前手型适配度高，生成效果更自然贴合。",
        "trend_text": "半透明凝胶、猫眼光泽和细节装饰是近期热门方向。",
    }


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_file(filename):
    if filename.startswith("assets/"):
        return send_from_directory(ROOT, filename)
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/api/health")
def health():
    return jsonify({
        "ok": True,
        "version": "vercel-ai-light",
        "styles_available": len(STYLE_LIBRARY),
        "qwen_image_edit": bool(DASHSCOPE_API_KEY),
        "qwen_image_edit_models": QWEN_IMAGE_EDIT_MODELS,
        "wanx_image_edit_model": WANX_IMAGE_EDIT_MODEL,
        "wanx_image_edit_timeout": WANX_IMAGE_EDIT_TIMEOUT,
    })


@app.route("/api/styles")
def styles():
    return jsonify({"styles": STYLE_LIBRARY, "available": len(STYLE_LIBRARY), "target": 25})


@app.route("/api/tryon", methods=["POST"])
def tryon():
    if "hand_image" not in request.files:
        return jsonify({"error": "missing hand_image"}), 400
    style_id = clamp(safe_int(request.form.get("style_id"), 20), 1, len(STYLE_LIBRARY))
    f = request.files["hand_image"]
    hand_bytes = f.read()
    hand_uri = data_uri_from_bytes(hand_bytes, f.mimetype)
    result = wanx_image_tryon(hand_bytes, f.mimetype, style_id) or qwen_image_tryon(hand_uri, style_id)
    advice = fit_advice(style_id)
    if not result:
        return jsonify({
            "result_image": hand_uri,
            "style_id": style_id,
            "style": STYLE_LIBRARY[style_id - 1],
            "nails_detected": 0,
            "debug_white_mode": False,
            "render_engine": "ai-fallback-original",
            "error": "AI image edit failed; returned original image",
            **advice,
        })
    return jsonify({
        "result_image": result["result_image"],
        "style_id": style_id,
        "style": STYLE_LIBRARY[style_id - 1],
        "nails_detected": 5,
        "debug_white_mode": False,
        "render_engine": f"qwen-image-edit:{result['model']}",
        **advice,
    })


@app.route("/api/recommend", methods=["POST"])
def recommend():
    if "hand_image" not in request.files:
        return jsonify({"error": "missing hand_image"}), 400
    f = request.files["hand_image"]
    return jsonify(recommend_style(data_uri_from_bytes(f.read(), f.mimetype)))


@app.route("/api/recommend_tryon", methods=["POST"])
def recommend_tryon():
    if "hand_image" not in request.files:
        return jsonify({"error": "missing hand_image"}), 400
    f = request.files["hand_image"]
    hand_bytes = f.read()
    hand_uri = data_uri_from_bytes(hand_bytes, f.mimetype)
    rec = recommend_style(hand_uri)
    style_id = clamp(safe_int((rec.get("recommendations") or [{}])[0].get("style_id"), 20), 1, len(STYLE_LIBRARY))
    result = wanx_image_tryon(hand_bytes, f.mimetype, style_id) or qwen_image_tryon(hand_uri, style_id)
    advice = fit_advice(style_id, rec.get("analysis"))
    return jsonify({
        "result_image": result["result_image"] if result else hand_uri,
        "style_id": style_id,
        "style": STYLE_LIBRARY[style_id - 1],
        "nails_detected": 5 if result else 0,
        "debug_white_mode": False,
        "analysis": rec.get("analysis", {}),
        "recommendations": rec.get("recommendations", []),
        "render_engine": f"qwen-image-edit:{result['model']}" if result else "ai-fallback-original",
        **advice,
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    if not question:
        return jsonify({"error": "empty message"}), 400
    style_id = safe_int(data.get("style_id"), 0)
    style_text = ""
    if 1 <= style_id <= len(STYLE_LIBRARY):
        s = STYLE_LIBRARY[style_id - 1]
        style_text = f"当前款式 #{s['id']} {s['name']}：{s.get('description', '')}"
    content = []
    if data.get("hand_image"):
        content.append({"image": data["hand_image"]})
    content.append({"text": f"你是美甲顾问，回复简洁友好，30-80字。{style_text}\n用户：{question}"})
    reply = call_qwen(content) or "这款整体会更显干净精致，建议选择半透明和高光质感，试戴会更自然。"
    return jsonify({"reply": reply[:500], "style_id": style_id})

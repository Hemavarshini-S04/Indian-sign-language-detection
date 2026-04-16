from flask import Flask, render_template, request, url_for, send_from_directory, Response, jsonify
from collections import Counter
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
from ultralytics.utils.plotting import Annotator
import cv2
import datetime
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import google.generativeai as genai
from google.api_core import exceptions as api_exceptions
import json
from pathlib import Path
import threading
import concurrent.futures
import secrets

# Try to load a local .env file if python-dotenv is installed. This is optional
# and won't raise if the package isn't available; it simply helps local testing.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

app = Flask(__name__)

# Folder configurations
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
REPORT_FOLDER = 'reports'
OUTPUT_FOLDER = 'static/outputs'

# Ensure directories exist
for folder in [UPLOAD_FOLDER, REPORT_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Load YOLO model for Indian Sign Language
model_path = "best.pt"  # Hypothetical model file
model = YOLO(model_path)

# Configure Gemini client from environment variable for safety
# Support either GENAI_API_KEY (preferred) or GOOGLE_API_KEY (some SDKs expect this name)
GENAI_API_KEY = os.getenv('GENAI_API_KEY') or os.getenv('GOOGLE_API_KEY')
if GENAI_API_KEY:
    # Ensure SDK sees GOOGLE_API_KEY env var as some libraries expect that name
    os.environ.setdefault('GOOGLE_API_KEY', GENAI_API_KEY)
    try:
        genai.configure(api_key=GENAI_API_KEY)
    except Exception as e:
        print(f"Warning: failed to configure google.generativeai: {e}")
else:
    print(
        "Warning: no Gemini API key found. Set GENAI_API_KEY or GOOGLE_API_KEY, or configure ADC. "
        "Gemini calls will be skipped or fallback will be used."
    )

# Simple in-memory circuit breaker for Gemini calls
GEMINI_DISABLED_UNTIL = 0
GEMINI_COOLDOWN_SECONDS = 300  # cooldown when quota errors occur (seconds)

# Persistent state to survive process restarts (stores disabled_until)
GEMINI_STATE_PATH = Path(__file__).parent / '.gemini_state.json'
_gemini_state_lock = threading.Lock()

# Admin token file for protecting runtime key set endpoint
ADMIN_TOKEN_PATH = Path(__file__).parent / '.admin_token'
ADMIN_TOKEN = None


def _init_admin_token():
    global ADMIN_TOKEN
    # Priority: environment variable (preconfigured) -> existing file -> generate new token
    env_token = os.getenv('ADMIN_TOKEN')
    if env_token:
        ADMIN_TOKEN = env_token
        return
    try:
        if ADMIN_TOKEN_PATH.exists():
            ADMIN_TOKEN = ADMIN_TOKEN_PATH.read_text(encoding='utf-8').strip()
            if ADMIN_TOKEN:
                return
    except Exception:
        pass
    # generate a strong random token and persist it
    ADMIN_TOKEN = secrets.token_urlsafe(24)
    try:
        ADMIN_TOKEN_PATH.write_text(ADMIN_TOKEN, encoding='utf-8')
        print(f"Admin token generated and saved to {ADMIN_TOKEN_PATH}. Token (showing once): {ADMIN_TOKEN}")
    except Exception:
        print("Warning: failed to persist admin token to file; token printed above only.")


# initialize admin token at startup
_init_admin_token()


def _load_gemini_state():
    global GEMINI_DISABLED_UNTIL
    try:
        if GEMINI_STATE_PATH.exists():
            with GEMINI_STATE_PATH.open('r', encoding='utf-8') as f:
                data = json.load(f)
            GEMINI_DISABLED_UNTIL = float(data.get('disabled_until', 0))
    except Exception:
        # ignore errors reading state
        GEMINI_DISABLED_UNTIL = GEMINI_DISABLED_UNTIL


def _save_gemini_state():
    try:
        data = {'disabled_until': GEMINI_DISABLED_UNTIL}
        with _gemini_state_lock:
            with GEMINI_STATE_PATH.open('w', encoding='utf-8') as f:
                json.dump(data, f)
    except Exception:
        # ignore write errors
        pass


# load persisted state at startup
_load_gemini_state()

# Email configurations
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_FROM = 'your-email@gmail.com'
EMAIL_PASSWORD = 'your-email-password'
EMAIL_TO = 'prosdgunal@gmail.com'

def send_email_alert(sign_list, report_filename, report_path):
    """Send an email alert with detection results and report attachment."""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO
        msg['Subject'] = 'Indian Sign Language Detection Alert'

        body = f"""
        Indian Sign Language Activity Detected!

        Detected Signs: {', '.join(sign_list)}

        A detailed report is attached for your reference.
        """
        msg.attach(MIMEText(body, 'plain'))

        with open(report_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())

        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename= {report_filename}'
        )
        msg.attach(part)

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        server.quit()
        print("Email alert sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

def process_image(results, model, image_path):
    """Process image detection results and save annotated image."""
    output_path = os.path.join(OUTPUT_FOLDER, 'output_image.jpg')
    image = cv2.imread(image_path)
    for r in results:
        annotator = Annotator(image)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
        img = annotator.result()
        cv2.imwrite(output_path, img)
    print(f"Image saved to {output_path}")
    return 'outputs/output_image.jpg'

def process_video(video_path, model):
    """Process video detection results and save annotated video with reduced FPS."""
    output_path = os.path.join(OUTPUT_FOLDER, 'output_video.mp4')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_fps = 5  # Set reduced FPS for slower video processing

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (frame_width, frame_height))

    sign_list = []
    frame_count = 0
    frame_interval = max(1, int(input_fps / output_fps))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        results = model.predict(frame)
        annotator = Annotator(frame)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls
                annotator.box_label(b, model.names[int(c)])
                sign_list.append(model.names[int(c)])

        annotated_frame = annotator.result()
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"Video saved to {output_path}")
    return 'outputs/output_video.mp4', list(set(sign_list))

def analyze_with_gemini(image_path, sign_list=None, max_retries=3, initial_backoff=1.0):
    """Analyze image using Gemini-2.0-flash model with retry/backoff and graceful fallback.

    This function implements:
    - a simple in-memory cooldown/circuit-breaker when quota is exhausted,
    - exponential backoff retries for transient errors,
    - a safe fallback string when analysis cannot be obtained.
    """
    global GEMINI_DISABLED_UNTIL

    # If we previously detected quota exhaustion, skip calling Gemini until cooldown expires
    if time.time() < GEMINI_DISABLED_UNTIL:
        return "Gemini analysis temporarily disabled due to quota/exhaustion."

    # If no API key was configured and no ADC, avoid calling the SDK (it will raise).
    # Provide a helpful message and local fallback using detected labels if available.
    if not GENAI_API_KEY:
        msg = (
            "Gemini analysis failed: No API key or ADC found. "
            "Set the environment variable `GENAI_API_KEY` or `GOOGLE_API_KEY`, or configure Application Default Credentials (ADC)."
        )
        if sign_list:
            msg += f" Local fallback summary: Detected signs: {', '.join(sign_list)}."
        else:
            msg += " Local fallback summary: No detected signs available."
        return msg

    # Do NOT send local file paths to the remote Gemini model (it cannot access local files).
    # Instead, if we have detected sign labels, ask Gemini to interpret those labels
    # and infer context/meaning. If no labels are available, request general guidance.
    if sign_list:
        prompt = (
            "Interpret the following detected Indian Sign Language labels and infer likely meaning, "
            f"grammatical role, and possible contexts: {', '.join(sign_list)}. "
            "Note: the actual image file is not accessible; use only the labels provided. "
            "Provide possible translations, usage notes, and suggestions for verifying the interpretation."
        )
    else:
        prompt = (
            "I cannot access the image (it is on a local filesystem). "
            "Please provide general guidance on how to analyze an Indian Sign Language image: "
            "what handshapes, locations, movements, and facial markers to look for, and "
            "questions to ask the user to help disambiguate signs when the image cannot be shared."
        )

    # Use a thread executor to enforce per-call timeouts (protects the request thread
    # from hanging if the remote API is unresponsive). We'll do up to `max_retries` attempts
    # but cap the total waiting time so user requests don't block indefinitely.
    backoff = initial_backoff
    max_total_wait = 15.0  # seconds total waiting across retries
    total_wait = 0.0
    per_call_timeout = 5.0  # seconds to wait for each SDK call

    def _call_genai(prompt_text):
        # This function runs in worker thread and calls the SDK.
        if hasattr(genai, 'GenerativeModel'):
            model_obj = genai.GenerativeModel('gemini-2.5-flash-lite')
            return model_obj.generate_content(prompt_text)
        if hasattr(genai, 'generate'):
            return genai.generate(model='gemini-2.5-flash-lite', input=prompt_text)
        if hasattr(genai, 'generate_text'):
            return genai.generate_text(model='gemini-2.5-flash-lite', prompt=prompt_text)
        raise RuntimeError('No supported google.generativeai call available')

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for attempt in range(1, max_retries + 1):
            # Check cooldown state (may have been set by previous runs)
            if time.time() < GEMINI_DISABLED_UNTIL:
                return "Gemini analysis temporarily disabled due to quota/exhaustion."

            future = executor.submit(_call_genai, prompt)
            try:
                response = future.result(timeout=per_call_timeout)
                if hasattr(response, 'text'):
                    return response.text
                return str(response)

            except concurrent.futures.TimeoutError:
                future.cancel()
                err = f"Gemini call timed out on attempt {attempt} (timeout={per_call_timeout}s)"
                print(err)
                total_wait += per_call_timeout
            except api_exceptions.ResourceExhausted as e:
                print(f"Gemini quota exhausted (attempt {attempt}): {e}")
                GEMINI_DISABLED_UNTIL = time.time() + GEMINI_COOLDOWN_SECONDS
                _save_gemini_state()
                return "Gemini analysis unavailable due to API quota limits."
            except Exception as e:
                print(f"Gemini API error on attempt {attempt}: {e}")
                total_wait += per_call_timeout

            # Decide whether to continue retrying
            if total_wait >= max_total_wait or attempt == max_retries:
                # Set cooldown to avoid repeated failures hammering the API
                GEMINI_DISABLED_UNTIL = time.time() + GEMINI_COOLDOWN_SECONDS
                _save_gemini_state()
                fallback = f"Gemini analysis failed after {attempt} attempts/timeouts."
                if sign_list:
                    fallback += f" Local fallback summary: Detected signs: {', '.join(sign_list)}."
                else:
                    fallback += " Local fallback summary: No detected signs available."
                return fallback

            # Sleep with exponential backoff but ensure we don't exceed max_total_wait
            sleep_time = min(backoff, max_total_wait - total_wait)
            time.sleep(sleep_time)
            total_wait += sleep_time
            backoff *= 2

def run_object_detection(file_path, is_video=False):
    """Run object detection on image or video."""
    if is_video:
        output_path, sign_list = process_video(file_path, model)
        print("Detected signs:", sign_list)
    else:
        results = model.predict(file_path)
        sign_counts = Counter(model.names[int(c)] for r in results for c in r.boxes.cls)
        sign_list = list(sign_counts.keys())
        print("Detected signs:", sign_list)
        output_path = process_image(results, model, file_path)

    gemini_analysis = analyze_with_gemini(file_path, sign_list=sign_list)
    print("Gemini Analysis:", gemini_analysis)

    return sign_list, output_path, gemini_analysis

def gen_frames():
    """Generate frames for live camera feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results[0].names[int(box.cls[0])]
            confidence = round(box.conf[0].item(), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def landing():
    """Render landing page."""
    return render_template('landing.html')

@app.route('/index')
def upload():
    """Render upload page."""
    return render_template('index.html')

@app.route('/live')
def live():
    """Render live camera page."""
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    """Stream live camera feed."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_file', methods=['POST'])
def process_file():
    """Process uploaded file and run detection."""
    if 'file' not in request.files:
        return render_template('result.html', error="No file part")

    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', error="No selected file")

    allowed_image_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    allowed_video_extensions = {'mp4', 'avi', 'mov'}
    filename = secure_filename(file.filename)
    file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

    if file_ext not in allowed_image_extensions and file_ext not in allowed_video_extensions:
        return render_template('result.html', error="Invalid file type")

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    is_video = file_ext in allowed_video_extensions
    sign_list, output_path, gemini_analysis = run_object_detection(file_path, is_video)
    if not output_path:
        return render_template('result.html', error="Error processing video")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{os.path.splitext(filename)[0]}_{timestamp}_report.txt"
    report_path = os.path.join(REPORT_FOLDER, report_filename)
    # Write report using UTF-8 to avoid UnicodeEncodeError on Windows (cp1252)
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(
            f"Indian Sign Language Detection Report\n\nDetected Signs: {', '.join(sign_list)}\n\nTimestamp: {timestamp}\n\nGemini Analysis: {gemini_analysis}"
        )

    send_email_alert(sign_list, report_filename, report_path)

    return render_template(
        'result.html',
        filename=filename,
        sign_list=sign_list,
        output_path=output_path,
        is_video=is_video,
        report_filename=report_filename,
        gemini_analysis=gemini_analysis
    )

@app.route('/static/outputs/<filename>')
def outputs(filename):
    """Serve output files."""
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/download_report/<filename>')
def download_report(filename):
    """Download report file."""
    return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)


@app.route('/debug/genai')
def debug_genai():
    """Return basic Gemini/GenAI diagnostics (no external calls).

    - shows whether an API key is configured (masked)
    - shows which SDK entry points appear present
    """
    key = os.getenv('GENAI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    masked = None
    if key:
        # mask all but last 4 chars
        masked = ('*' * max(0, len(key) - 4)) + key[-4:]

    sdk_info = {
        'has_GenerativeModel': hasattr(genai, 'GenerativeModel'),
        'has_generate': hasattr(genai, 'generate'),
        'has_generate_text': hasattr(genai, 'generate_text'),
    }

    return {
        'api_key_present': bool(key),
        'api_key_masked': masked,
        'sdk_info': sdk_info,
        'genie_warning': 'Warning: no Gemini API key found. Set GENAI_API_KEY or GOOGLE_API_KEY, or configure ADC.' if not key else None,
    }


@app.route('/admin')
def admin_page():
    """Render a minimal admin page to set GENAI_API_KEY (uses the POST endpoint)."""
    return render_template('admin.html')


def _update_env_file(var_name: str, var_value: str):
    """Update (or create) a .env file in the project root with the given variable.

    This updates an existing entry if present, otherwise appends a new line.
    """
    env_path = Path(__file__).parent / '.env'
    lines = []
    try:
        if env_path.exists():
            lines = env_path.read_text(encoding='utf-8').splitlines()
        found = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{var_name}="):
                lines[i] = f"{var_name}={var_value}"
                found = True
                break
        if not found:
            lines.append(f"{var_name}={var_value}")
        env_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return True
    except Exception as e:
        print(f"Warning: failed to write .env file: {e}")
        return False


@app.route('/admin/set_genai_key', methods=['POST'])
def admin_set_genai_key():
    """Admin endpoint to set/persist GENAI_API_KEY at runtime.

    Expects JSON: {"key": "<API_KEY>", "token": "<ADMIN_TOKEN>"}
    The endpoint updates the running process environment, calls genai.configure,
    and persists the key to `.env` for future restarts.
    """
    data = request.get_json(silent=True) or {}
    key = data.get('key')
    token = data.get('token')
    if not key or not token:
        return jsonify({'ok': False, 'error': 'Missing key or token in request (JSON).'}), 400
    # validate token
    if ADMIN_TOKEN is None or token != ADMIN_TOKEN:
        return jsonify({'ok': False, 'error': 'Invalid admin token.'}), 403

    # persist to .env and set runtime environment
    ok = _update_env_file('GENAI_API_KEY', key)
    if not ok:
        return jsonify({'ok': False, 'error': 'Failed to persist .env file.'}), 500

    os.environ['GENAI_API_KEY'] = key
    os.environ['GOOGLE_API_KEY'] = key
    try:
        genai.configure(api_key=key)
    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to configure genai: {e}'}), 500

    return jsonify({'ok': True, 'message': 'GENAI_API_KEY updated and persisted.'})

if __name__ == '__main__':
    app.run(debug=True)
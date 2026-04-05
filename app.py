from flask import Flask, render_template_string

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Smart Air Whiteboard — Teacher Edition</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
  :root{
    --bg:#0d0d14;--card:#13131f;--border:#1e1e2e;
    --accent:#00e5c8;--accent2:#7c6cf7;--text:#e8e8f0;--muted:#6b6b80;
  }
  *{margin:0;padding:0;box-sizing:border-box;}
  body{background:var(--bg);color:var(--text);font-family:'Space Grotesk',sans-serif;min-height:100vh;overflow-x:hidden;}

  /* Grid background */
  body::before{
    content:'';position:fixed;inset:0;
    background-image:linear-gradient(var(--border) 1px,transparent 1px),
                     linear-gradient(90deg,var(--border) 1px,transparent 1px);
    background-size:60px 60px;opacity:.5;pointer-events:none;
  }

  header{
    text-align:center;padding:80px 20px 40px;
    position:relative;
  }
  .badge{
    display:inline-block;background:rgba(0,229,200,.1);
    border:1px solid var(--accent);color:var(--accent);
    font-family:'JetBrains Mono',monospace;font-size:.7rem;
    letter-spacing:.15em;padding:4px 14px;border-radius:4px;
    margin-bottom:20px;
  }
  h1{
    font-size:clamp(2.2rem,6vw,4.2rem);font-weight:700;line-height:1.1;
    background:linear-gradient(135deg,#fff 30%,var(--accent));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    margin-bottom:16px;
  }
  .subtitle{color:var(--muted);font-size:1.1rem;max-width:540px;margin:0 auto 40px;}

  .pill-row{display:flex;flex-wrap:wrap;gap:10px;justify-content:center;margin-bottom:60px;}
  .pill{
    background:rgba(124,108,247,.12);border:1px solid rgba(124,108,247,.3);
    color:#a89ff5;font-size:.8rem;padding:5px 14px;border-radius:20px;
    font-family:'JetBrains Mono',monospace;
  }

  /* Feature grid */
  .grid{
    display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
    gap:20px;max-width:1100px;margin:0 auto;padding:0 24px 60px;
  }
  .card{
    background:var(--card);border:1px solid var(--border);
    border-radius:16px;padding:28px;
    transition:border-color .2s,transform .2s;
    position:relative;overflow:hidden;
  }
  .card::before{
    content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,var(--accent2),var(--accent));
    opacity:0;transition:opacity .2s;
  }
  .card:hover{border-color:rgba(0,229,200,.25);transform:translateY(-3px);}
  .card:hover::before{opacity:1;}
  .card-icon{font-size:2rem;margin-bottom:14px;}
  .card h3{font-size:1rem;font-weight:600;margin-bottom:8px;color:var(--accent);}
  .card p{font-size:.875rem;color:var(--muted);line-height:1.6;}

  /* Gesture table */
  .section{max-width:860px;margin:0 auto 60px;padding:0 24px;}
  .section-title{
    font-size:.7rem;letter-spacing:.2em;color:var(--accent);
    font-family:'JetBrains Mono',monospace;text-transform:uppercase;
    margin-bottom:20px;
  }
  table{width:100%;border-collapse:collapse;}
  th{
    text-align:left;font-size:.75rem;color:var(--muted);
    font-family:'JetBrains Mono',monospace;letter-spacing:.1em;
    padding:8px 16px;border-bottom:1px solid var(--border);
  }
  td{padding:12px 16px;border-bottom:1px solid rgba(30,30,46,.6);font-size:.9rem;}
  tr:hover td{background:rgba(255,255,255,.02);}
  .gest{
    font-family:'JetBrains Mono',monospace;font-size:.8rem;
    background:rgba(124,108,247,.12);color:#a89ff5;
    padding:3px 10px;border-radius:6px;white-space:nowrap;
  }

  /* Install block */
  .install{
    max-width:860px;margin:0 auto 80px;padding:0 24px;
  }
  .code-block{
    background:#090912;border:1px solid var(--border);border-radius:12px;
    padding:24px;font-family:'JetBrains Mono',monospace;font-size:.82rem;
    line-height:2;position:relative;overflow:auto;
  }
  .code-block .comment{color:var(--muted);}
  .code-block .cmd{color:var(--accent);}
  .code-block .arg{color:#f9c74f;}

  /* Theme demo */
  .themes{display:flex;flex-wrap:wrap;gap:16px;max-width:860px;margin:0 auto 60px;padding:0 24px;}
  .theme-box{
    flex:1;min-width:180px;border-radius:12px;padding:20px;
    border:1px solid var(--border);text-align:center;
  }
  .theme-box .dot{width:40px;height:40px;border-radius:50%;margin:0 auto 10px;}
  .theme-box p{font-size:.8rem;color:var(--muted);}
  .theme-box strong{font-size:.95rem;}

  footer{
    text-align:center;padding:40px 20px;border-top:1px solid var(--border);
    color:var(--muted);font-size:.8rem;
  }
  footer span{color:var(--accent);}

  /* Floating orbs */
  .orb{
    position:fixed;border-radius:50%;filter:blur(80px);
    opacity:.12;pointer-events:none;z-index:0;
  }
  .orb1{width:500px;height:500px;background:var(--accent2);top:-200px;right:-200px;}
  .orb2{width:400px;height:400px;background:var(--accent);bottom:-150px;left:-150px;}
</style>
</head>
<body>
<div class="orb orb1"></div>
<div class="orb orb2"></div>

<header>
  <div class="badge">v4.0 TEACHER EDITION</div>
  <h1>Smart Air Whiteboard</h1>
  <p class="subtitle">Draw, write &amp; teach in mid-air using just your hands — powered by MediaPipe &amp; OpenCV</p>
  <div class="pill-row">
    <span class="pill">☝ Gesture Drawing</span>
    <span class="pill">🔍 OCR Text Recognition</span>
    <span class="pill">🎨 3 Themes</span>
    <span class="pill">✏️ 7 Tools</span>
    <span class="pill">😊 Emotion Detection</span>
    <span class="pill">⏱️ Class Timer</span>
    <span class="pill">20-Level Undo</span>
    <span class="pill">Auto-Save</span>
  </div>
</header>

<!-- Features -->
<div class="grid">
  <div class="card">
    <div class="card-icon">✍️</div>
    <h3>7 Drawing Tools</h3>
    <p>Marker, Chalk (rough texture), Highlighter (transparent), Laser Pointer, Symmetry, Radial, and Constellation mode.</p>
  </div>
  <div class="card">
    <div class="card-icon">🔍</div>
    <h3>Auto OCR</h3>
    <p>Write something, pause for 2 seconds — the app automatically reads your handwriting using Tesseract OCR and shows the recognized text.</p>
  </div>
  <div class="card">
    <div class="card-icon">🎨</div>
    <h3>3 Board Themes</h3>
    <p>Whiteboard (white + dark ink), Dark Board (neon glow), and Blackboard (green + chalk). Switch with the <code>W</code> key.</p>
  </div>
  <div class="card">
    <div class="card-icon">🧠</div>
    <h3>Kalman Smoothing</h3>
    <p>Finger trajectory is mathematically smoothed using a Kalman filter — lines are silky-smooth, no jitter.</p>
  </div>
  <div class="card">
    <div class="card-icon">😊</div>
    <h3>Emotion Detection</h3>
    <p>Face mesh landmarks detect if the teacher looks Happy, Sad, Surprised or Neutral — shown in the HUD.</p>
  </div>
  <div class="card">
    <div class="card-icon">⏱️</div>
    <h3>Class Timer</h3>
    <p>Built-in 5-minute countdown timer. Turns red when time is almost up. Toggle with the <code>T</code> key anytime.</p>
  </div>
  <div class="card">
    <div class="card-icon">✨</div>
    <h3>Particle Effects</h3>
    <p>Drawing leaves a trail of glowing particles that fade with gravity — making each stroke feel alive.</p>
  </div>
  <div class="card">
    <div class="card-icon">💾</div>
    <h3>Smart Save</h3>
    <p>Saves the board as a composite image (background + drawing) — exactly as it looks on screen. Auto-saves every 60 seconds.</p>
  </div>
</div>

<!-- Gesture table -->
<div class="section">
  <div class="section-title">// Gesture Controls</div>
  <table>
    <tr><th>GESTURE</th><th>ACTION</th><th>HOW TO</th></tr>
    <tr><td><span class="gest">☝ Index only</span></td><td>Draw / Write</td><td>Extend only index finger</td></tr>
    <tr><td><span class="gest">✌ Peace sign</span></td><td>Erase</td><td>Extend index + middle</td></tr>
    <tr><td><span class="gest">✊ Fist (hold 1s)</span></td><td>Clear board</td><td>Close all fingers, hold 1 second</td></tr>
    <tr><td><span class="gest">🖐 Open palm</span></td><td>Pause / Resume</td><td>Open all fingers</td></tr>
    <tr><td><span class="gest">👍 Thumbs up</span></td><td>Save snapshot</td><td>Thumb up, others closed</td></tr>
    <tr><td><span class="gest">🤟 Rock sign</span></td><td>Cycle tool</td><td>Index + pinky extended</td></tr>
    <tr><td><span class="gest">3 fingers</span></td><td>Cycle color</td><td>Index + middle + ring</td></tr>
  </table>
</div>

<!-- Themes -->
<div class="section">
  <div class="section-title">// Board Themes</div>
</div>
<div class="themes">
  <div class="theme-box" style="background:#f0f0eb;border-color:#ddd;">
    <div class="dot" style="background:#000;"></div>
    <strong style="color:#222;">Whiteboard</strong>
    <p style="color:#666;">White bg + dark ink</p>
  </div>
  <div class="theme-box" style="background:#12121a;">
    <div class="dot" style="background:#00e5c8;"></div>
    <strong style="color:#e0e0e0;">Dark Board</strong>
    <p>Dark bg + neon glow</p>
  </div>
  <div class="theme-box" style="background:#1c3426;">
    <div class="dot" style="background:#ffdc50;"></div>
    <strong style="color:#f0f0dc;">Blackboard</strong>
    <p style="color:#aaa;">Green bg + chalk texture</p>
  </div>
</div>

<!-- Install -->
<div class="install">
  <div class="section-title">// Run Locally</div>
  <div class="code-block">
<span class="comment"># 1. Install dependencies</span>
<span class="cmd">pip install</span> <span class="arg">-r requirements.txt</span>

<span class="comment"># 2. Install Tesseract OCR engine</span>
<span class="cmd">brew install</span> <span class="arg">tesseract</span>         <span class="comment"># macOS</span>
<span class="cmd">sudo apt install</span> <span class="arg">tesseract-ocr</span>   <span class="comment"># Ubuntu/Linux</span>

<span class="comment"># 3. Run</span>
<span class="cmd">python</span> <span class="arg">smart_whiteboard.py</span>

<span class="comment"># Keyboard shortcuts</span>
<span class="comment"># W → Cycle theme | R → OCR scan | T → Timer</span>
<span class="comment"># Z → Undo | C → Clear | S → Save | Q → Quit</span>
<span class="comment"># 1-7 → Quick tool select</span>
  </div>
</div>

<footer>
  Built with <span>MediaPipe · OpenCV · Tesseract · Python</span><br><br>
  Smart Air Whiteboard v4.0 — Teacher Edition
</footer>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/health")
def health():
    return {"status": "ok", "app": "Smart Air Whiteboard", "version": "4.0"}

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

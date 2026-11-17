import hashlib
import hmac
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

st.set_page_config(page_title="Plant Transcription Pattern Detector | NanStack", page_icon="🌱",
                   layout="wide", initial_sidebar_state="collapsed")

DEV_NAME, DEV_URL = "NanStack", "https://NanStack.tech"
# Credentials are stored as SHA-256 hashes (not plain text)
USER_HASH = "2ff163b53c27502187017a43c6b0635f544da3c65d6c1abaacf7b1bd6d9d31d0"
PASS_HASH = "b48e75697b1986eeae5c86797e4f57de9451762b5e82e82cf684ab9f227ed1ae"
MAX_ATTEMPTS, LOCK_SECONDS = 5, 30

ACCENT, ACCENT2, TEXT, MUTED = "#34d399", "#22d3ee", "#e6f4ee", "#8fa9a0"

# ----------------------------------------------------------------------------- styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp {
  background: radial-gradient(1200px 600px at 10% -10%, rgba(52,211,153,.16), transparent 60%),
              radial-gradient(1000px 600px at 100% 0%, rgba(34,211,238,.12), transparent 55%),
              #07110e;
  color: #e6f4ee;
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2rem; max-width: 1200px; }
h1, h2, h3, h4, p, label, span, div { color: inherit; }

.hero { text-align: center; margin: 1.2rem 0 1.6rem 0; }
.hero .badge { display:inline-block; padding:.35rem .9rem; border-radius:999px; font-size:.78rem; font-weight:600;
  letter-spacing:.08em; color:#34d399; background:rgba(52,211,153,.10); border:1px solid rgba(52,211,153,.35); }
.hero h1 { font-size:2.6rem; font-weight:800; margin:.8rem 0 .4rem 0; line-height:1.15;
  background:linear-gradient(90deg,#34d399,#22d3ee); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p { color:#8fa9a0; font-size:1.02rem; max-width:640px; margin:0 auto; }

[data-testid="stForm"] { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.10);
  border-radius:20px; padding:1.8rem 1.8rem 1.2rem 1.8rem; backdrop-filter: blur(12px);
  box-shadow:0 20px 60px rgba(0,0,0,.45); }
.stTextInput input, .stTextArea textarea { background:rgba(255,255,255,.05) !important; color:#e6f4ee !important;
  border:1px solid rgba(255,255,255,.14) !important; border-radius:12px !important; }
.stTextInput input:focus, .stTextArea textarea:focus { border-color:#34d399 !important; box-shadow:0 0 0 2px rgba(52,211,153,.25) !important; }

.stButton > button, [data-testid="stFormSubmitButton"] > button { width:100%; border-radius:12px; font-weight:600; padding:.6rem 1rem;
  border:1px solid rgba(255,255,255,.16); background:rgba(255,255,255,.05); color:#e6f4ee; transition:all .2s ease; }
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { border-color:#34d399; color:#34d399; transform:translateY(-1px); }
[data-testid="stFormSubmitButton"] > button[kind="primary"], .stButton > button[kind="primary"] {
  background:linear-gradient(90deg,#10b981,#06b6d4); color:#04120d; border:none; }
[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover { color:#04120d; filter:brightness(1.1); }

.divider { display:flex; align-items:center; gap:.8rem; color:#8fa9a0; font-size:.8rem; margin:.9rem 0; }
.divider:before, .divider:after { content:""; flex:1; height:1px; background:rgba(255,255,255,.12); }

.card { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.09); border-radius:16px; padding:1.1rem 1.2rem; }
.kpi .label { font-size:.74rem; letter-spacing:.08em; text-transform:uppercase; color:#8fa9a0; }
.kpi .value { font-size:1.7rem; font-weight:800; margin-top:.2rem; background:linear-gradient(90deg,#34d399,#22d3ee);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.chip { display:inline-block; margin:.2rem .3rem .2rem 0; padding:.3rem .8rem; border-radius:999px; font-size:.82rem; font-weight:600;
  color:#34d399; background:rgba(52,211,153,.10); border:1px solid rgba(52,211,153,.35); }
.topbar { display:flex; justify-content:space-between; align-items:center; margin-bottom:.4rem; }
.topbar .brand { font-weight:800; font-size:1.05rem; }
.userpill { padding:.3rem .8rem; border-radius:999px; font-size:.8rem; color:#8fa9a0; border:1px solid rgba(255,255,255,.14); }

.stTabs [data-baseweb="tab-list"] { gap:.4rem; border-bottom:1px solid rgba(255,255,255,.08); }
.stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; padding:.6rem 1.1rem; color:#8fa9a0; }
.stTabs [aria-selected="true"] { color:#34d399 !important; }

.dev-footer { text-align:center; margin:2.6rem 0 .6rem 0; padding-top:1.2rem; border-top:1px solid rgba(255,255,255,.08); color:#8fa9a0; font-size:.88rem; }
.dev-footer a { display:inline-block; margin-left:.35rem; padding:.28rem .8rem; border-radius:999px; font-weight:700; text-decoration:none;
  color:#04120d !important; background:linear-gradient(90deg,#34d399,#22d3ee); transition:all .2s ease; }
.dev-footer a:hover { transform:translateY(-2px); box-shadow:0 8px 24px rgba(52,211,153,.35); }
[data-testid="stSidebar"] { background:#09160f; border-right:1px solid rgba(255,255,255,.08); }

.kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.8rem; }
.kpi .value { word-break:break-word; }
.stTextInput input, .stTextArea textarea { font-size:16px !important; }
[data-testid="stDataFrame"] { overflow-x:auto; }

@media (max-width: 900px) {
  .kpi-grid { grid-template-columns:repeat(2,1fr); }
}
@media (max-width: 640px) {
  .block-container { padding:1rem .8rem 2rem .8rem !important; }
  .hero { margin:.6rem 0 1rem 0; }
  .hero h1 { font-size:1.75rem !important; }
  .hero p { font-size:.92rem; }
  [data-testid="stForm"] { padding:1.2rem 1rem .8rem 1rem; border-radius:16px; }
  .kpi .value { font-size:1.2rem; }
  .card { padding:.8rem .9rem; }
  .stTabs [data-baseweb="tab"] { padding:.5rem .6rem; font-size:.85rem; }
  .stTabs [data-baseweb="tab-list"] { overflow-x:auto; }
  .dev-footer { margin-top:1.8rem; }
  [data-testid="stHorizontalBlock"] { gap:.5rem !important; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def dev_footer():
    st.markdown(
        f'<div class="dev-footer">Developed by <a href="{DEV_URL}" target="_blank" rel="noopener noreferrer">{DEV_NAME} ↗</a>'
        f'<div style="margin-top:.5rem;font-size:.75rem;opacity:.7">© 2026 {DEV_NAME}. All rights reserved.</div></div>',
        unsafe_allow_html=True)


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- auth
st.session_state.setdefault("role", None)       # None | "admin" | "guest"
st.session_state.setdefault("attempts", 0)
st.session_state.setdefault("locked_until", 0.0)


def login_page():
    st.markdown(
        '<div class="hero"><span class="badge">🌱 PLANT GENOMICS · DEEP LEARNING</span>'
        '<h1>Plant Transcription<br>Pattern Detector</h1>'
        '<p>AI-based detection of transcription factor motifs from Arabidopsis promoter sequences. Please sign in to continue.</p></div>',
        unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        remaining = st.session_state.locked_until - time.time()
        with st.form("login_form"):
            st.markdown("#### 🔐 Sign in")
            user = st.text_input("Username", placeholder="Enter username")
            pwd = st.text_input("Password", type="password", placeholder="Enter password")
            submit = st.form_submit_button("Login", type="primary")
        if submit:
            if remaining > 0:
                st.error(f"Too many failed attempts. Please try again in {int(remaining) + 1} seconds.")
            else:
                ok = hmac.compare_digest(sha(user), USER_HASH) & hmac.compare_digest(sha(pwd), PASS_HASH)
                if ok:
                    st.session_state.role, st.session_state.attempts = "admin", 0
                    st.rerun()
                else:
                    st.session_state.attempts += 1
                    left = MAX_ATTEMPTS - st.session_state.attempts
                    if left <= 0:
                        st.session_state.locked_until = time.time() + LOCK_SECONDS
                        st.session_state.attempts = 0
                        st.error(f"Too many failed attempts. Login is locked for {LOCK_SECONDS} seconds.")
                    else:
                        st.error(f"Incorrect username or password. Attempts remaining: {left}")

        st.markdown('<div class="divider">or</div>', unsafe_allow_html=True)
        if st.button("👤  Continue as Guest"):
            st.session_state.role = "guest"
            st.rerun()
        st.caption("Guest mode includes predictions and visualizations. Custom model upload is available to signed-in users only.")
    dev_footer()


if st.session_state.role is None:
    login_page()
    st.stop()

is_admin = st.session_state.role == "admin"

# ----------------------------------------------------------------------------- model
class PromoterCNN(nn.Module):
    def __init__(self, n_labels, seq_len=1000):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 128, kernel_size=15, padding=7), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(0.2),
            nn.Conv1d(128, 128, kernel_size=9, padding=4), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(0.2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (seq_len // 16), 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_labels),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


@st.cache_resource
def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = PromoterCNN(len(ck["labels"]), ck["seq_len"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    pwms = [np.array(p, dtype=np.float32) for p in ck["pwms"]]
    return m, ck, pwms


LUT = {"A": 0, "C": 1, "G": 2, "T": 3}


def clean(text):
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith(">")]
    return "".join(lines).upper().replace(" ", "")


def encode(seq):
    return np.array([LUT[c] for c in seq], dtype=np.int64)


def onehot(idx):
    return F.one_hot(torch.from_numpy(idx), 4).permute(0, 2, 1).float()


def pwm_hits(idx, pwm, thr):
    L = pwm.shape[0]
    oh = np.eye(4, dtype=np.float32)[idx]
    win = sliding_window_view(oh, (L, 4))[:, 0]
    mn, mx = pwm.min(1).sum(), pwm.max(1).sum()
    rc = pwm[::-1, [3, 2, 1, 0]]
    rel_f = ((win * pwm).sum((1, 2)) - mn) / (mx - mn)
    rel_r = ((win * rc).sum((1, 2)) - mn) / (mx - mn)
    out = []
    for s in np.where(np.maximum(rel_f, rel_r) >= thr)[0]:
        out.append((int(s), int(s + L), "+" if rel_f[s] >= rel_r[s] else "-", float(max(rel_f[s], rel_r[s]))))
    return out


def style_ax(ax):
    ax.set_facecolor("none")
    for sp in ax.spines.values():
        sp.set_color("#2a4036")
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED); ax.title.set_color(TEXT)


def kpi_grid(items):
    cells = "".join(f'<div class="card kpi"><div class="label">{l}</div><div class="value">{v}</div></div>' for l, v in items)
    st.markdown(f'<div class="kpi-grid">{cells}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- top bar + sidebar
c1, c2, c3 = st.columns([5, 2, 1])
c1.markdown('<div class="topbar"><span class="brand">🌱 Plant Transcription Pattern Detector</span></div>', unsafe_allow_html=True)
c2.markdown(f'<div style="text-align:right"><span class="userpill">{"🔑 NanStack (Admin)" if is_admin else "👤 Guest"}</span></div>',
            unsafe_allow_html=True)
if c3.button("Logout"):
    st.session_state.role = None
    st.rerun()

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    threshold = st.slider("Prediction threshold", 0.05, 0.95, 0.5, 0.05)
    path = "plant_motif_cnn.pt"
    if is_admin:
        up = st.file_uploader("Custom model (.pt)", type=["pt"])
        if up is not None:
            with open("uploaded_model.pt", "wb") as f:
                f.write(up.getbuffer())
            path = "uploaded_model.pt"
    else:
        st.info("Custom model upload is available to signed-in users only.")
    st.markdown(f'<div class="dev-footer" style="margin-top:2rem">Developed by '
                f'<a href="{DEV_URL}" target="_blank" rel="noopener noreferrer">{DEV_NAME} ↗</a></div>', unsafe_allow_html=True)

try:
    model, ck, pwms = load_model(path)
except Exception as e:
    st.error("Could not load the model. Place `plant_motif_cnn.pt` in the same folder as app.py.")
    st.exception(e)
    dev_footer()
    st.stop()

labels, SEQ_LEN, REL_THR = ck["labels"], ck["seq_len"], ck["rel_thr"]
names = [l.split("|")[0] for l in labels]

# ----------------------------------------------------------------------------- hero + input
st.markdown('<div class="hero" style="margin-bottom:1rem"><span class="badge">AI · PROMOTER ANALYSIS</span>'
            '<h1 style="font-size:2rem">Analyze a promoter sequence</h1>'
            f'<p>Paste an Arabidopsis promoter sequence of at least {SEQ_LEN} bp (FASTA format is also accepted).</p></div>',
            unsafe_allow_html=True)

st.session_state.setdefault("seq", "")
if st.button("✨ Random example (planted W-box TTGAC)"):
    rng = np.random.default_rng(0)
    s = list("".join(rng.choice(list("ACGT"), SEQ_LEN)))
    s[700:705] = list("TTGAC")
    st.session_state.seq = "".join(s)
    st.rerun()

text = st.text_area("Promoter sequence", value=st.session_state.seq, height=150,
                    placeholder="ACGT... or FASTA with a > header", label_visibility="collapsed")
st.session_state.seq = text
seq = clean(text)

if not seq:
    st.info("Paste a sequence or click the example button.")
    dev_footer()
    st.stop()
if set(seq) - set("ACGT"):
    st.error("The sequence may only contain A, C, G and T.")
    dev_footer()
    st.stop()
if len(seq) < SEQ_LEN:
    st.error(f"The sequence is {len(seq)} bp long; at least {SEQ_LEN} bp is required.")
    dev_footer()
    st.stop()

idx = encode(seq)
starts = list(range(0, len(idx) - SEQ_LEN + 1, 250))
if starts[-1] != len(idx) - SEQ_LEN:
    starts.append(len(idx) - SEQ_LEN)

with torch.no_grad():
    probs = np.stack([torch.sigmoid(model(onehot(idx[s:s + SEQ_LEN][None]))).numpy()[0] for s in starts])
best = probs.max(0)
detected = [names[i] for i in range(len(names)) if best[i] >= threshold]

kpi_grid([("Sequence length", f"{len(seq)} bp"), ("Windows scanned", len(starts)),
          ("Families detected", f"{len(detected)} / {len(names)}"), ("Top prediction", names[int(best.argmax())])])
st.write("")

tab1, tab2, tab3 = st.tabs(["📊 Predictions", "🧬 Motif & Saliency", "ℹ️ About"])

with tab1:
    if detected:
        st.markdown("**Detected:** " + "".join(f'<span class="chip">{d}</span>' for d in detected), unsafe_allow_html=True)
    else:
        st.write("No family was detected at this threshold.")
    a, b = st.columns([1, 1])
    df = pd.DataFrame({"TF family": names, "Motif": [l.split("|")[1] for l in labels],
                       "Probability": best.round(3), "Detected": best >= threshold})
    a.dataframe(df.sort_values("Probability", ascending=False), hide_index=True)
    fig, ax = plt.subplots(figsize=(5.2, 4))
    fig.patch.set_alpha(0)
    order = np.argsort(best)
    ax.barh(np.array(names)[order], best[order], color=[ACCENT if v >= threshold else "#2f4a40" for v in best[order]])
    ax.axvline(threshold, color="#f87171", ls="--", lw=1)
    ax.set_xlim(0, 1); ax.set_xlabel("Probability"); style_ax(ax)
    b.pyplot(fig, transparent=True)
    if len(starts) > 1:
        st.caption(f"The sequence is {len(seq)} bp long and was scanned in {len(starts)} windows (1000 bp each). The maximum score per family is shown.")

with tab2:
    ca, cb = st.columns(2)
    k = ca.selectbox("TF family", range(len(labels)), format_func=lambda i: f"{names[i]} ({labels[i].split('|')[1]})")
    w = 0
    if len(starts) > 1:
        w = cb.selectbox("Window", range(len(starts)), format_func=lambda i: f"{starts[i]}-{starts[i] + SEQ_LEN} bp")
    widx = idx[starts[w]:starts[w] + SEQ_LEN]
    x = onehot(widx[None]).requires_grad_(True)
    model.zero_grad()
    model(x)[0, k].backward()
    sal = (x.grad * x).sum(1)[0].detach().numpy()
    hits = pwm_hits(widx, pwms[k], REL_THR)

    fig2, ax2 = plt.subplots(figsize=(11, 3.2))
    fig2.patch.set_alpha(0)
    ax2.plot(sal, lw=1.1, color=ACCENT2)
    for s, e, _, _ in hits:
        ax2.axvspan(s, e, color="#fbbf24", alpha=0.35)
    ax2.axvline(ck.get("up", 800), color="#f87171", ls="--", label="TSS")
    ax2.set_xlabel("Position (bp)"); ax2.set_ylabel("Saliency")
    ax2.set_title(f"{names[k]}: saliency (gradient x input), orange = PWM motif hits")
    leg = ax2.legend(facecolor="none", edgecolor="#2a4036")
    for t in leg.get_texts():
        t.set_color(TEXT)
    style_ax(ax2)
    st.pyplot(fig2, transparent=True)

    if hits:
        st.dataframe(pd.DataFrame([{"start": s + starts[w], "end": e + starts[w], "strand": sd, "score": round(sc, 3),
                                    "sequence": seq[s + starts[w]:e + starts[w]]} for s, e, sd, sc in hits]),
                     hide_index=True)
    else:
        st.write("No strong motif hit for this family was found in this window.")

with tab3:
    st.markdown(f"""
<div class="card">
<h4>About this project</h4>
<p>This app uses a 1D CNN that predicts transcription factor family motifs from Arabidopsis thaliana promoter sequences
(800 bp upstream + 200 bp downstream of the TSS). Saliency maps show which bases the model focused on.</p>
<p><b>Model labels:</b> weak labels generated from JASPAR plant PWM scans. Use DAP-seq data for real binding validation.</p>
<p>Developed by <a href="{DEV_URL}" target="_blank" rel="noopener noreferrer" style="color:#34d399;font-weight:700">{DEV_NAME}</a></p>
</div>""", unsafe_allow_html=True)

dev_footer()
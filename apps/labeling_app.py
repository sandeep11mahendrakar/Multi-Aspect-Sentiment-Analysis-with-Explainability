"""E7A labeling UI - blind human annotation of rating-3 reviews.

Run from repo root:
    streamlit run apps/labeling_app.py

Score scale 0..9 (buttons + keyboard):
    0-3 -> negative | 4-6 -> neutral | 7-9 -> positive
Keyboard: Left/Right arrows = prev/next review, digits 0-9 = assign score
and advance. Scores save to CSV instantly. Star ratings stay hidden.
"""

import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E7A_DIR = os.path.join(REPO_ROOT, "results", "e7a")
ROUNDS = {
    "Round 1 (200 reviews)": os.path.join(E7A_DIR, "e7a_labeling_round1.csv"),
    "Round 2 (50 reviews, kappa subset)": os.path.join(E7A_DIR, "e7a_labeling_round2.csv"),
}
SENTENCES_CSV = os.path.join(E7A_DIR, "e7a_sentences_view.csv")
ASPECT_EMOJI = {"quality": "🧵", "price": "💰", "fit": "📏", "shipping": "🚚"}


def bucket(score: int) -> str:
    return "negative" if score <= 3 else ("positive" if score >= 7 else "neutral")


BUCKET_COLOR = {"negative": "#ef4444", "neutral": "#9ca3af", "positive": "#10b981"}

st.set_page_config(page_title="E7A Labeling", page_icon="🏷️", layout="centered")


@st.cache_data(ttl=5)
def load_sentences():
    if os.path.exists(SENTENCES_CSV):
        return pd.read_csv(SENTENCES_CSV).fillna("")
    return None


def load_round(path):
    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    if "score" not in df.columns:
        df["score"] = ""
    return df


def save_and_move(df, path, idx, value):
    df.iloc[idx, df.columns.get_loc("score")] = value
    df.to_csv(path, index=False, encoding="utf-8-sig")
    st.cache_data.clear()
    st.session_state["idx"] = min(idx + 1, len(df) - 1)
    st.rerun()


round_name = st.sidebar.selectbox("Labeling round", list(ROUNDS))
csv_path = ROUNDS[round_name]
df = load_round(csv_path)
sentences = load_sentences()

done = int((df["score"] != "").sum())
st.sidebar.metric("Labeled", f"{done} / {len(df)}")
st.sidebar.progress(done / len(df))
st.sidebar.markdown("**Scale:** 0–3 negative · 4–6 neutral · 7–9 positive")
st.sidebar.markdown("⌨️ ←/→ = prev/next · keys **0–9** = score")

if st.session_state.get("active_round") != round_name:
    st.session_state["active_round"] = round_name
    st.session_state["idx"] = 0
if "idx" not in st.session_state:
    st.session_state["idx"] = 0

col_a, col_b, col_c = st.sidebar.columns(3)
if col_a.button("◀ Prev"):
    st.session_state["idx"] = max(0, st.session_state["idx"] - 1)
if col_b.button("Next ▶"):
    st.session_state["idx"] = min(len(df) - 1, st.session_state["idx"] + 1)
if col_c.button("⏭ Todo"):
    unlabeled = df.index[df["score"] == ""].tolist()
    nxt = [i for i in unlabeled if i > st.session_state["idx"]]
    st.session_state["idx"] = nxt[0] if nxt else (unlabeled[0] if unlabeled else st.session_state["idx"])

idx = min(st.session_state["idx"], len(df) - 1)
row = df.iloc[idx]
rid = int(row["review_id"])

st.markdown(f"### Review {idx + 1} of {len(df)}  ·  id `{rid}`")
if str(row["score"]) != "":
    s = int(row["score"])
    b = bucket(s)
    st.caption(f"current score: **{s}** ({b}) — rescore to overwrite")

# ---- sentence view with aspect tags ----
sent_df = sentences[sentences["review_id"] == rid] if sentences is not None else None
if sent_df is not None and len(sent_df):
    for _, srow in sent_df.iterrows():
        tags = "".join(
            f" {ASPECT_EMOJI.get(a, '🏷️')} `{a}`" for a in str(srow["aspects"]).split(",") if a
        )
        st.markdown(f"- {srow['sentence']}{tags}")
else:
    st.write(row["review_text"])

st.divider()

# ---- score buttons 0..9 ----
legend = "  ".join(
    f"<span style='color:{BUCKET_COLOR[bucket(v)]};font-size:0.72rem'>"
    f"{'neg' if v == 0 else 'neu' if v == 4 else 'pos' if v == 7 else '&nbsp;'}</span>"
    for v in range(10))
st.markdown(f"<div style='text-align:center'>{legend}</div>", unsafe_allow_html=True)

btn_cols = st.columns(10)
for v, col in enumerate(btn_cols):
    if col.button(str(v), key=f"score_{v}", use_container_width=True):
        save_and_move(df, csv_path, idx, v)

st.caption("⌨️ ← → navigate · 0–9 score & advance · Tab = next")

# ---- keyboard shortcuts (arrows + digits + Tab fallback) ----
# Key actions are queued and only dispatched when Streamlit is idle, so rapid
# keypresses never land on a detached DOM node mid-rerun. The handler is bound
# to BOTH the parent document and the component iframe itself, because focus
# can end up in either one after Streamlit re-renders.
components.html("""
<script>
(function(){
  // Defer until this iframe is actually attached to the live page. Running
  // earlier means window.parent.document belongs to a detached tree and any
  // bindings made on it silently never see real events.
  let bound = false;
  function init(){
    if (bound) return;
    let fe = null;
    try { fe = window.frameElement; } catch (e) { fe = null; }
    if (!fe || !fe.isConnected) { setTimeout(init, 60); return; }
    const d = window.parent.document;
    if (!d.contains(fe)) { setTimeout(init, 60); return; }
    bound = true;
    console.log('[e7a] iframe attached; binding keyboard layer');
    const state = d.__e7a_state || (d.__e7a_state = { q: [], pumping: false });

  const appBusy = function(){
    if (d.querySelector('[data-testid="stStatusWidget"]')) return true;
    const app = d.querySelector('.stApp');
    return !!(app && app.getAttribute('aria-busy') === 'true');
  };
  const findBtn = function(txt){
    return Array.from(d.querySelectorAll('button')).find(function(b){
      return !b.disabled && b.offsetParent !== null &&
             (b.textContent || '').trim() === txt;
    });
  };
  const pump = function(){
    if (!state.q.length) { state.pumping = false; return; }
    if (appBusy()) { setTimeout(pump, 120); return; }
    const txt = state.q.shift();
    let tries = 0;
    const attempt = function(){
      if (appBusy()) { setTimeout(attempt, 120); return; }
      const b = findBtn(txt);
      if (b) {
        b.click();
        setTimeout(pump, 300);   // let the rerun start before next action
      } else if (++tries < 25) {
        setTimeout(attempt, 120); // DOM still re-rendering, retry
      } else {
        pump();
      }
    };
    attempt();
  };
  const enqueue = function(txt){
    state.q.push(txt);
    if (!state.pumping) { state.pumping = true; setTimeout(pump, 40); }
  };

  const onKeyDown = function(e){
    const t = e.target;
    const tag = (t && t.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    let txt = null;
    if (e.key === 'ArrowRight' || e.key === 'Tab') txt = 'Next \u25B6';
    else if (e.key === 'ArrowLeft') txt = '\u25C0 Prev';
    else if (/^[0-9]$/.test(e.key)) txt = e.key;
    if (txt) { e.preventDefault(); enqueue(txt); }
  };

  // parent document: install once
  if (!d.__e7a_keys_installed) {
    d.__e7a_keys_installed = true;
    d.addEventListener('keydown', onKeyDown, true);
  }
  // this iframe's document is fresh on every Streamlit rerun: always bind
  document.addEventListener('keydown', onKeyDown, true);
  }
  init();
})();
</script>
""", height=0)

if done == len(df):
    st.success("All reviews labeled! Run: venv\\Scripts\\python.exe src\\experiments\\analyze_e7a.py")

import re
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import easyocr

st.set_page_config(page_title="Game Screenshot Analyzer", page_icon="📊", layout="wide")
st.title("📊 Game Screenshot Analyzer")
st.caption("Upload screenshots only. OCR extracts visible history; check the extracted rows before analysis. This tool does not predict or guarantee future results.")

@st.cache_resource
def get_reader():
    return easyocr.Reader(["en"], gpu=False)

def read_image(img):
    reader = get_reader()
    arr = np.array(img.convert("RGB"))
    return reader.readtext(arr, detail=0, paragraph=False)

def clean_period(s):
    return re.sub(r"[^0-9]", "", str(s))

def extract_candidates(lines, game):
    text = "\\n".join(lines)
    candidates = []
    # OCR line-by-line; retain source text so user can correct uncertain reads.
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        if game == "aviator":
            vals = re.findall(r"(?<!\\d)(\\d{1,3}(?:[.,]\\d{1,2})?)\\s*[xX]?", raw)
            for v in vals:
                try:
                    n = float(v.replace(",", "."))
                    if 1 <= n <= 100000:
                        candidates.append({"period": "", "result": f"{n:.2f}x", "source_text": raw})
                except ValueError:
                    pass
        elif game == "big_small":
            for v in re.findall(r"\\b(BIG|SMALL)\\b", raw, flags=re.I):
                candidates.append({"period": "", "result": v.title(), "source_text": raw})
        elif game == "colour":
            for v in re.findall(r"\\b(RED|GREEN|VIOLET|PURPLE)\\b", raw, flags=re.I):
                candidates.append({"period": "", "result": v.title(), "source_text": raw})
        elif game == "k3":
            # Prefer standalone 1-3 digit values; zero-pad to preserve 000-099.
            for v in re.findall(r"(?<!\\d)(\\d{1,3})(?!\\d)", raw):
                candidates.append({"period": "", "result": v.zfill(3), "source_text": raw})
    # Remove exact duplicate candidates caused by OCR line repeats.
    seen, out = set(), []
    for c in candidates:
        key = (c["result"], c["source_text"])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out

def category(game, value):
    v = str(value).strip()
    if game == "big_small":
        return v.title()
    if game == "colour":
        return v.title()
    if game == "k3":
        return "Big" if int(v) >= 500 else "Small"
    n = float(v.lower().replace("x", "").replace(",", "."))
    if n < 2: return "Low (1.00–1.99x)"
    if n < 5: return "Medium (2.00–4.99x)"
    if n < 10: return "High (5.00–9.99x)"
    return "Very high (10x+)"

def analyze(df, game):
    d = df.copy()
    d["period_num"] = pd.to_numeric(d["period"], errors="coerce")
    if d["period_num"].notna().all() and len(d):
        d = d.sort_values("period_num")
    # Otherwise preserve upload/extraction order; user can reverse table if screenshot newest-first.
    labels = [category(game, x) for x in d["result"]]
    freq = pd.Series(labels).value_counts().rename_axis("Category").reset_index(name="Count")
    if len(labels):
        freq["Frequency %"] = (freq["Count"] / len(labels) * 100).round(1)
    transitions = pd.crosstab(pd.Series(labels[:-1], name="Current"),
                              pd.Series(labels[1:], name="Next"), normalize="index").round(3) if len(labels) > 1 else pd.DataFrame()
    best_len, best_cat, cur_len, cur_cat = 0, None, 0, None
    for lab in labels:
        if lab == cur_cat: cur_len += 1
        else: cur_cat, cur_len = lab, 1
        if cur_len > best_len: best_len, best_cat = cur_len, lab
    return d.drop(columns=["period_num"]), freq, transitions, (best_cat, best_len)

st.info("Choose the game shown in your screenshots, then upload one or more history screenshots. Different websites arrange history differently, so verify OCR before trusting the report.")
game = st.selectbox("Game type", ["K3 (3-digit)", "Big / Small", "Colour", "Aviator"])
game_key = {"K3 (3-digit)": "k3", "Big / Small": "big_small", "Colour": "colour", "Aviator": "aviator"}[game]
uploads = st.file_uploader("Upload history screenshots", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

if uploads:
    all_rows, ocr_blocks = [], []
    with st.spinner("Reading screenshots with OCR… first run may take a little longer while the OCR model loads."):
        for file in uploads:
            image = Image.open(file)
            lines = read_image(image)
            ocr_blocks.append((file.name, lines))
            rows = extract_candidates(lines, game_key)
            for row in rows:
                row["screenshot"] = file.name
                all_rows.append(row)

    st.subheader("1. Screenshot previews and OCR text")
    for name, lines in ocr_blocks:
        with st.expander(f"OCR text — {name}"):
            st.text("\\n".join(lines) if lines else "No readable text detected.")

    st.subheader("2. Check and correct extracted history")
    st.caption("OCR can confuse digits, period IDs and multipliers. Delete incorrect rows, correct results, and fill period numbers where visible. For K3, keep leading zeroes (e.g. 007).")
    if all_rows:
        initial = pd.DataFrame(all_rows)
    else:
        initial = pd.DataFrame(columns=["period", "result", "source_text", "screenshot"])
        st.warning("No clear results were extracted. Try a sharper screenshot with the history list enlarged.")
    edited = st.data_editor(initial, num_rows="dynamic", use_container_width=True, key="history_editor")
    valid = edited.dropna(subset=["result"]).copy()
    valid["result"] = valid["result"].astype(str).str.strip()
    valid = valid[valid["result"] != ""]

    def valid_result(v):
        try:
            if game_key == "k3": return bool(re.fullmatch(r"\\d{1,3}", v)) and 0 <= int(v) <= 999
            if game_key == "big_small": return v.lower() in {"big", "small"}
            if game_key == "colour": return v.lower() in {"red", "green", "violet", "purple"}
            return float(v.lower().replace("x", "").replace(",", ".")) >= 1
        except Exception:
            return False

    valid = valid[valid["result"].map(valid_result)]
    if len(valid):
        st.subheader("3. Historical analysis")
        data = valid[["period", "result"]].copy()
        data["period"] = data["period"].fillna("").astype(str)
        data["result"] = data["result"].astype(str)
        ordered, freq, trans, streak = analyze(data, game_key)
        c1, c2 = st.columns(2)
        c1.metric("Rounds analysed", len(ordered))
        c2.metric("Longest same-category streak", f"{streak[1]} ({streak[0] or '—'})")
        st.write("**Historical frequency**")
        st.dataframe(freq, use_container_width=True, hide_index=True)
        st.write("**Historical transition table** (descriptive only; not a future prediction)")
        if not trans.empty: st.dataframe(trans, use_container_width=True)
        else: st.caption("Need at least two results for transition analysis.")
        if game_key == "aviator":
            nums = pd.to_numeric(ordered["result"].str.lower().str.replace("x", "", regex=False).str.replace(",", ".", regex=False))
            st.write({"Mean multiplier": round(float(nums.mean()), 3), "Median": round(float(nums.median()), 3),
                      "At least 2x (%)": round(float((nums >= 2).mean()*100), 1),
                      "At least 5x (%)": round(float((nums >= 5).mean()*100), 1),
                      "At least 10x (%)": round(float((nums >= 10).mean()*100), 1)})
        csv = ordered.to_csv(index=False).encode("utf-8")
        st.download_button("Download cleaned history CSV", csv, "cleaned_game_history.csv", "text/csv")
        report = freq.to_csv(index=False).encode("utf-8")
        st.download_button("Download frequency report CSV", report, f"{game_key}_frequency.csv", "text/csv")
    else:
        st.warning("No valid rows yet. Correct the table above or add rows manually.")

st.divider()
st.caption("Important: Historical frequencies, streaks and transitions do not reveal or control future RNG outcomes. No next-10 winning predictions are generated. Use this for record-keeping and statistical learning, not betting decisions.")

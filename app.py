# ==========================================================
# app.py - Dashboard Prediksi Harga Saham ANTM (LSTM)
# Tugas Akhir - Nico Viogi Pratama
# Sumber data: FILE CSV TETAP (offline) agar reproducible & bebas rate-limit
# ==========================================================
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# ----------------------------------------------------------
# 1. KONFIGURASI (harus sama persis dengan notebook pelatihan)
# ----------------------------------------------------------
TICKER = "ANTM.JK"
NAMA_SAHAM = "PT Aneka Tambang Tbk"
LOOKBACK = 30
FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume", "Return", "HL_Range",
    "OC_Change", "MA5", "MA10", "MA20", "STD5", "RSI14", "Vol_Change",
]
MODEL_PATH = "model/ANTM_JK_lstm_model.keras"
SCALER_PATH = "model/ANTM_JK_feature_scaler.pkl"
# Dataset historis tetap (2021-06-30 s/d 2026-06-30). Sertakan file ini di repo.
DATA_PATH = "data/ANTM_JK_historical_ohlcv_5y.csv"
PRED_TEST_PATH = "data/ANTM_JK_lstm_test_predictions_open_close_nextday.csv"

st.set_page_config(page_title="Prediksi Saham ANTM", page_icon="📈", layout="wide")

# ----------------------------------------------------------
# 2. LOAD MODEL & SCALER (di-cache agar tidak diulang)
# ----------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# ----------------------------------------------------------
# 3. FUNGSI FITUR (identik dengan notebook)
# ----------------------------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(period).mean() / loss.rolling(period).mean().replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

@st.cache_data
def ambil_data():
    # Membaca dataset historis dari FILE CSV (bukan online) -> bebas rate-limit
    raw = pd.read_csv(DATA_PATH)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw[["Date", "Open", "High", "Low", "Close", "Volume"]].sort_values("Date").reset_index(drop=True)
    raw = raw[raw["Volume"] > 0].reset_index(drop=True)
    return raw

def bangun_fitur(raw):
    df = raw.copy()
    df["Return"]     = df["Close"].pct_change()
    df["HL_Range"]   = (df["High"] - df["Low"]) / df["Close"]
    df["OC_Change"]  = (df["Close"] - df["Open"]) / df["Open"]
    df["MA5"]        = df["Close"].rolling(5).mean()
    df["MA10"]       = df["Close"].rolling(10).mean()
    df["MA20"]       = df["Close"].rolling(20).mean()
    df["STD5"]       = df["Close"].rolling(5).std()
    df["RSI14"]      = compute_rsi(df["Close"], 14)
    df["Vol_Change"] = df["Volume"].pct_change()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return df

def prediksi_hari_berikutnya(model, scaler, df):
    X = scaler.transform(df[FEATURE_COLUMNS].values[-LOOKBACK:]).reshape(1, LOOKBACK, len(FEATURE_COLUMNS))
    logret = model.predict(X, verbose=0)[0]
    last_close = float(df["Close"].iloc[-1])
    last_date = pd.to_datetime(df["Date"].iloc[-1]).date()
    next_date = (pd.to_datetime(last_date) + pd.tseries.offsets.BDay(1)).date()
    pred_open = last_close * np.exp(logret[0])
    pred_close = last_close * np.exp(logret[1])
    return last_date, next_date, last_close, float(pred_open), float(pred_close)

# ----------------------------------------------------------
# 4. LOGIKA REKOMENDASI AKSI
# ----------------------------------------------------------
def rekomendasi(last_close, pred_open, pred_close, ambang=0.5):
    chg_open = (pred_open - last_close) / last_close * 100
    chg_close = (pred_close - last_close) / last_close * 100
    intraday = (pred_close - pred_open) / pred_open * 100

    open_naik = chg_open > ambang
    open_turun = chg_open < -ambang
    intraday_naik = intraday > ambang
    intraday_turun = intraday < -ambang

    if open_naik and intraday_naik:
        sinyal, warna = "BELI / HOLD (Bullish kuat)", "green"
        aksi = ("Harga diprediksi dibuka lebih tinggi dan terus naik hingga penutupan. "
                "Rekomendasi: **BELI hari ini menjelang penutupan** untuk mengambil posisi sebelum kenaikan, "
                "atau **HOLD** jika sudah memegang saham. Pertimbangkan merealisasikan keuntungan mendekati penutupan besok.")
    elif open_naik and intraday_turun:
        sinyal, warna = "JUAL di Pembukaan (Bearish intraday)", "orange"
        aksi = ("Harga diprediksi dibuka lebih tinggi namun melemah hingga penutupan. "
                "Rekomendasi: jika sudah memegang saham, **JUAL / ambil untung di sesi pembukaan besok** saat harga masih tinggi. "
                "**Hindari membeli di pembukaan** karena berisiko turun sepanjang hari.")
    elif open_turun and intraday_naik:
        sinyal, warna = "BELI di Pembukaan (Rebound intraday)", "green"
        aksi = ("Harga diprediksi dibuka lebih rendah lalu berbalik naik hingga penutupan. "
                "Rekomendasi: **BELI di sesi pembukaan besok saat harga mendekati titik terendah (open)**, "
                "kemudian pertimbangkan menjual mendekati penutupan untuk menangkap kenaikan intraday.")
    elif open_turun and intraday_turun:
        sinyal, warna = "JUAL / WAIT (Bearish kuat)", "red"
        aksi = ("Harga diprediksi dibuka lebih rendah dan terus turun hingga penutupan. "
                "Rekomendasi: **hindari membeli**; jika sudah memegang saham pertimbangkan **menjual hari ini** atau "
                "**menunggu (wait & see)** hingga tren membaik.")
    else:
        sinyal, warna = "HOLD / NETRAL", "gray"
        aksi = ("Perubahan harga yang diprediksi relatif kecil (di bawah ambang signifikan). "
                "Rekomendasi: **HOLD / tahan posisi** dan amati perkembangan pasar sebelum mengambil keputusan.")
    return sinyal, warna, aksi, chg_open, chg_close, intraday

# ==========================================================
# 5. TAMPILAN (UI)
# ==========================================================
st.title("📈 Dashboard Prediksi Harga Saham ANTM")
st.caption(f"{NAMA_SAHAM} ({TICKER}) — Model Long Short-Term Memory (LSTM) berbasis data historis")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    ambang = st.slider("Ambang sinyal signifikan (%)", 0.0, 3.0, 0.5, 0.1)
    st.markdown("---")
    st.info("Aplikasi ini memvisualisasikan output model LSTM. Bukan merupakan ajakan atau nasihat investasi.")

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Gagal memuat model/scaler. Pastikan file ada di folder 'model/'. Detail: {e}")
    st.stop()

try:
    raw = ambil_data()
except FileNotFoundError:
    st.error(f"File dataset tidak ditemukan: {DATA_PATH}. "
             "Unggah file CSV historis ANTM ke folder 'data/' pada repo Anda.")
    st.stop()

df = bangun_fitur(raw)
last_date, next_date, last_close, pred_open, pred_close = prediksi_hari_berikutnya(model, scaler, df)
sinyal, warna, aksi, chg_open, chg_close, intraday = rekomendasi(last_close, pred_open, pred_close, ambang)

# ---- Kartu ringkasan ----
st.subheader(f"Prediksi untuk Hari Perdagangan Berikutnya ({next_date})")
c1, c2, c3 = st.columns(3)
c1.metric(f"Close terakhir ({last_date})", f"Rp {last_close:,.0f}")
c2.metric("Prediksi Open", f"Rp {pred_open:,.2f}", f"{chg_open:+.2f}%")
c3.metric("Prediksi Close", f"Rp {pred_close:,.2f}", f"{chg_close:+.2f}%")

# ---- Kotak rekomendasi ----
st.subheader("💡 Rekomendasi Aksi")
warna_map = {"green": "#1a7f37", "orange": "#b3560f", "red": "#b42318", "gray": "#475467"}
st.markdown(
    f"""<div style='padding:16px;border-radius:10px;background:{warna_map[warna]}20;border-left:6px solid {warna_map[warna]}'>
    <h4 style='margin:0;color:{warna_map[warna]}'>{sinyal}</h4>
    <p style='margin:6px 0 0 0'>{aksi}</p>
    <p style='margin:8px 0 0 0;font-size:0.85em;color:#667085'>Perkiraan pergerakan open→close besok: {intraday:+.2f}%</p>
    </div>""",
    unsafe_allow_html=True,
)

# ---- Grafik harga historis + titik prediksi ----
st.subheader("Grafik Harga Historis dan Prediksi")
plot_df = df.tail(120)
fig = go.Figure()
fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Close"], name="Close historis", line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(x=[pd.to_datetime(next_date)], y=[pred_open], name="Prediksi Open", mode="markers",
                         marker=dict(color="#2ca02c", size=12, symbol="triangle-up")))
fig.add_trace(go.Scatter(x=[pd.to_datetime(next_date)], y=[pred_close], name="Prediksi Close", mode="markers",
                         marker=dict(color="#d62728", size=12, symbol="circle")))
fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Tanggal", yaxis_title="Harga (Rp)")
st.plotly_chart(fig, use_container_width=True)

# ---- (Opsional) Evaluasi pada data uji jika CSV tersedia ----
st.subheader("Evaluasi Model pada Data Uji (opsional)")
try:
    ev = pd.read_csv(PRED_TEST_PATH, parse_dates=["Target_Date"])
    def metrik(a, p):
        a, p = np.array(a, float), np.array(p, float)
        mae = np.mean(np.abs(a - p)); rmse = np.sqrt(np.mean((a - p) ** 2))
        mape = np.mean(np.abs((a - p) / a)) * 100
        ss = 1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)
        return mae, rmse, mape, ss
    mo = metrik(ev["Actual_Open"], ev["Predicted_Open"])
    mc = metrik(ev["Actual_Close"], ev["Predicted_Close"])
    tab = pd.DataFrame([["Open", *mo], ["Close", *mc]], columns=["Target", "MAE", "RMSE", "MAPE (%)", "R2"]).round(4)
    st.dataframe(tab, use_container_width=True)
    figev = go.Figure()
    figev.add_trace(go.Scatter(x=ev["Target_Date"], y=ev["Actual_Close"], name="Actual Close"))
    figev.add_trace(go.Scatter(x=ev["Target_Date"], y=ev["Predicted_Close"], name="Predicted Close"))
    figev.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(figev, use_container_width=True)
except FileNotFoundError:
    st.caption("Letakkan file prediksi data uji di folder 'data/' untuk menampilkan evaluasi (opsional).")

st.markdown("---")
st.caption("⚠️ Disclaimer: Prediksi bersifat probabilistik dan hanya untuk keperluan akademik/demonstrasi, "
           "bukan nasihat keuangan. Keputusan investasi sepenuhnya tanggung jawab pengguna.")

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
import yfinance as yf

# =========================================================
# CANAVAR AI TRADE TERMINAL v4.0
# Dip dönüşü + hedef fiyat + dinamik stop + backtest
# =========================================================

st.set_page_config(
    page_title="Canavar AI Trade Terminal v4.0",
    layout="wide",
    initial_sidebar_state="expanded",
)

st_autorefresh(interval=60_000, key="canavar_autorefresh")

DATA_DIR = Path("canavar_data")
DATA_DIR.mkdir(exist_ok=True)
PORTFOY_DOSYASI = DATA_DIR / "portfoy_data.json"
PIK_DOSYASI = DATA_DIR / "pik_fiyatlar.json"
ALARMLAR_DOSYASI = DATA_DIR / "alarmlar_data.json"
SINYAL_DOSYASI = DATA_DIR / "sinyal_gecmisi.json"

# Tokenı doğrudan koda yazmayın.
# Windows PowerShell örneği:
#   $env:TELEGRAM_TOKEN="..."
#   $env:TELEGRAM_CHAT_ID="..."
# Alternatif: .streamlit/secrets.toml içine ekleyin.
def _secret_veya_env(anahtar: str, varsayilan: str = "") -> str:
    try:
        return str(st.secrets.get(anahtar, os.getenv(anahtar, varsayilan)))
    except Exception:
        return os.getenv(anahtar, varsayilan)


TELEGRAM_TOKEN = _secret_veya_env("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = _secret_veya_env("TELEGRAM_CHAT_ID")

BIST_OTOMATIK_HAVUZ = [
    "AGHOL", "AKBNK", "AKCNS", "AKSEN", "ALARK", "ALBRK", "ALFAS", "ARCLK",
    "ASELS", "ASTOR", "BIMAS", "BRSAN", "BRYAT", "BUCIM", "CCOLA", "CIMSA",
    "CWENE", "DOAS", "DOHOL", "ECILC", "EGEEN", "EKGYO", "ENJSA", "ENKAI",
    "EREGL", "EUPWR", "FROTO", "GARAN", "GESAN", "GUBRF", "HALKB", "HEKTS",
    "IMASM", "IPEKE", "ISCTR", "ISGYO", "ISMEN", "IZMDC", "KARDMD", "KCHOL",
    "KONTR", "KONYA", "KOZAA", "KOZAL", "KCAER", "MAVI", "MGROS", "MIATK",
    "ODAS", "OTKAR", "OYAKC", "PETKM", "PGSUS", "QUAGR", "REEDR", "SAHOL",
    "SASA", "SAYAS", "SISE", "SKBNK", "SMRTG", "SOKM", "TABGD", "TARKM",
    "TCELL", "THYAO", "TKFEN", "TOASO", "TSKB", "TTKOM", "TUPRS", "TURSG",
    "ULKER", "VAKBN", "VESTL", "YEOTK", "YKBNK", "ZOREN",
]


@dataclass
class TradePlan:
    ticker: str
    fiyat: float
    dip_puani: int
    sinyal_guveni: int
    asama: str
    alim_alt: float
    alim_ust: float
    hedef_1: float
    hedef_2: float
    stop: float
    potansiyel_1: float
    potansiyel_2: float
    risk_yuzde: float
    risk_kazanc: float
    atr: float
    atr_yuzde: float
    rsi: float
    gunluk_degisim: float
    tahmini_sure: str
    nedenler: List[str]
    uyarilar: List[str]


# -------------------------
# Dosya ve bildirim yardımcıları
# -------------------------
def veri_yukle(dosya_adi: Path, varsayilan: Any) -> Any:
    if not dosya_adi.exists():
        return varsayilan
    try:
        with dosya_adi.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return varsayilan


def veri_kaydet(dosya_adi: Path, veri: Any) -> None:
    gecici = dosya_adi.with_suffix(dosya_adi.suffix + ".tmp")
    try:
        with gecici.open("w", encoding="utf-8") as f:
            json.dump(veri, f, ensure_ascii=False, indent=2, default=str)
        gecici.replace(dosya_adi)
    except OSError as exc:
        st.error(f"Dosya kaydetme hatası: {exc}")


def telegram_bildirim_gonder(mesaj: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": mesaj}
    try:
        yanit = requests.post(url, data=payload, timeout=8)
        yanit.raise_for_status()
        return True
    except requests.RequestException:
        return False


def tr_fiyat(x: float) -> str:
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# -------------------------
# Piyasa verisi
# -------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fiyat_verisi_getir(ticker_name: str, period: str = "2y") -> pd.DataFrame:
    """Yahoo Finance verisini birkaç yöntemle dener; geçici ağ/API hatalarında boş dönmeyi azaltır."""
    sembol = f"{ticker_name}.IS"
    gerekli = ["Open", "High", "Low", "Close", "Volume"]

    denemeler = [
        {"period": period, "interval": "1d"},
        {"period": "1y", "interval": "1d"},
        {"period": "6mo", "interval": "1d"},
    ]
    for ayar in denemeler:
        try:
            df = yf.download(
                sembol,
                period=ayar["period"],
                interval=ayar["interval"],
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=15,
            )
            if isinstance(df.columns, pd.MultiIndex):
                # Tek sembolde yfinance sürümüne göre kolon sırası değişebilir.
                if all(c in df.columns.get_level_values(0) for c in gerekli):
                    df.columns = df.columns.get_level_values(0)
                else:
                    df.columns = df.columns.get_level_values(-1)
            if not df.empty and all(c in df.columns for c in gerekli):
                df = df[gerekli].dropna(subset=["Close"]).copy()
                if len(df) >= 55:
                    return df
        except Exception:
            pass

    try:
        df = yf.Ticker(sembol).history(period=period, interval="1d", auto_adjust=True)
        if not df.empty and all(c in df.columns for c in gerekli):
            return df[gerekli].dropna(subset=["Close"]).copy()
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def temel_veri_getir(ticker_name: str) -> Dict[str, Any]:
    try:
        return dict(yf.Ticker(f"{ticker_name}.IS").info or {})
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def haberler_getir(ticker_name: str) -> List[Dict[str, Any]]:
    try:
        return list(yf.Ticker(f"{ticker_name}.IS").news or [])[:10]
    except Exception:
        return []


def guncel_fiyat_bul(ticker_name: str) -> Optional[float]:
    df = fiyat_verisi_getir(ticker_name, "1mo")
    if df.empty:
        return None
    return round(float(df["Close"].iloc[-1]), 2)


# -------------------------
# Teknik göstergeler
# -------------------------
def gostergeleri_hesapla(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    close, high, low, volume = x["Close"], x["High"], x["Low"], x["Volume"]

    x["SMA10"] = close.rolling(10).mean()
    x["SMA20"] = close.rolling(20).mean()
    x["SMA50"] = close.rolling(50).mean()
    x["SMA100"] = close.rolling(100).mean()
    x["SMA200"] = close.rolling(200).mean()
    x["EMA9"] = close.ewm(span=9, adjust=False).mean()
    x["EMA20"] = close.ewm(span=20, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["RSI"] = (100 - 100 / (1 + rs)).fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    x["MACD"] = ema12 - ema26
    x["MACD_SIGNAL"] = x["MACD"].ewm(span=9, adjust=False).mean()
    x["MACD_HIST"] = x["MACD"] - x["MACD_SIGNAL"]

    std20 = close.rolling(20).std()
    x["BB_UPPER"] = x["SMA20"] + 2 * std20
    x["BB_LOWER"] = x["SMA20"] - 2 * std20
    x["BB_WIDTH"] = (x["BB_UPPER"] - x["BB_LOWER"]) / x["SMA20"].replace(0, np.nan)

    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    x["STOCH_K"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    x["STOCH_D"] = x["STOCH_K"].rolling(3).mean()

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    x["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    x["ATR_PCT"] = 100 * x["ATR"] / close.replace(0, np.nan)

    x["VOL_MA10"] = volume.rolling(10).mean()
    x["VOL_MA20"] = volume.rolling(20).mean()
    x["VOL_RATIO"] = volume / x["VOL_MA20"].replace(0, np.nan)
    x["ROC5"] = close.pct_change(5) * 100
    x["ROC20"] = close.pct_change(20) * 100
    x["LOW20"] = low.rolling(20).min()
    x["LOW50"] = low.rolling(50).min()
    x["HIGH20"] = high.rolling(20).max()
    x["HIGH50"] = high.rolling(50).max()
    return x.replace([np.inf, -np.inf], np.nan)


def pivot_seviyeleri(series: pd.Series, pencere: int = 3) -> Tuple[List[float], List[float]]:
    destekler, direncler = [], []
    vals = series.dropna().to_numpy(dtype=float)
    if len(vals) < 2 * pencere + 1:
        return destekler, direncler
    for i in range(pencere, len(vals) - pencere):
        bolum = vals[i - pencere : i + pencere + 1]
        if vals[i] == np.min(bolum):
            destekler.append(float(vals[i]))
        if vals[i] == np.max(bolum):
            direncler.append(float(vals[i]))
    return destekler, direncler


def yakin_seviyeleri_birlestir(seviyeler: List[float], tolerans: float = 0.02) -> List[float]:
    if not seviyeler:
        return []
    sonuc: List[float] = []
    for s in sorted(seviyeler):
        if not sonuc or abs(s - sonuc[-1]) / max(sonuc[-1], 1e-9) > tolerans:
            sonuc.append(s)
        else:
            sonuc[-1] = (sonuc[-1] + s) / 2
    return sonuc


def destek_direnc_bul(df: pd.DataFrame) -> Tuple[List[float], List[float]]:
    son = df.tail(180)
    destek_low, _ = pivot_seviyeleri(son["Low"], 3)
    _, direnc_high = pivot_seviyeleri(son["High"], 3)
    destekler = yakin_seviyeleri_birlestir(destek_low + [float(son["Low"].rolling(20).min().iloc[-1])])
    direncler = yakin_seviyeleri_birlestir(direnc_high + [float(son["High"].rolling(20).max().iloc[-1])])
    return destekler, direncler


def haber_duygu_skori(ticker_name: str) -> Tuple[int, str]:
    haberler = haberler_getir(ticker_name)
    if not haberler:
        return 0, "Aktif haber bulunamadı"

    pozitif = {
        "ihale": 2, "sözleşme": 2, "sozlesme": 2, "yatırım": 1, "yatirim": 1,
        "temettü": 1, "temettu": 1, "bedelsiz": 1, "rekor": 1, "büyüme": 1,
        "buyume": 1, "ihracat": 1, "ortaklık": 1, "ortaklik": 1, "kâr": 1,
        "kar": 1, "geri alım": 1, "geri alim": 1,
    }
    negatif = {
        "iptal": -2, "ceza": -2, "zarar": -1, "dava": -1, "borç": -1,
        "borc": -1, "temerrüt": -2, "temerrut": -2, "soruşturma": -2,
        "sorusturma": -2, "satış": -1, "satis": -1,
    }
    skor = 0
    for haber in haberler[:5]:
        baslik = str(haber.get("title", "")).lower()
        for kelime, puan in pozitif.items():
            if kelime in baslik:
                skor += puan
        for kelime, puan in negatif.items():
            if kelime in baslik:
                skor += puan
    skor = int(max(-5, min(5, skor)))
    aciklama = "Pozitif haber akışı" if skor > 0 else "Negatif haber akışı" if skor < 0 else "Nötr haber akışı"
    return skor, aciklama


# -------------------------
# Türkçe temel analiz özeti
# -------------------------
SEKTOR_TR = {
    "Financial Services": "Finansal hizmetler",
    "Industrials": "Sanayi",
    "Technology": "Teknoloji",
    "Consumer Cyclical": "Döngüsel tüketim",
    "Consumer Defensive": "Temel tüketim",
    "Basic Materials": "Temel malzemeler",
    "Energy": "Enerji",
    "Utilities": "Altyapı ve enerji hizmetleri",
    "Healthcare": "Sağlık",
    "Communication Services": "İletişim hizmetleri",
    "Real Estate": "Gayrimenkul",
}

ENDUSTRI_TR = {
    "Banks—Regional": "Bölgesel bankacılık",
    "Auto Manufacturers": "Otomotiv üretimi",
    "Airlines": "Havayolu taşımacılığı",
    "Steel": "Demir-çelik",
    "Oil & Gas Refining & Marketing": "Petrol rafinajı ve pazarlaması",
    "Telecom Services": "Telekomünikasyon",
    "Conglomerates": "Holding ve iştirak yönetimi",
    "Building Materials": "Yapı malzemeleri",
    "Grocery Stores": "Gıda perakendeciliği",
    "Electrical Equipment & Parts": "Elektrik ekipmanları",
    "Specialty Industrial Machinery": "Özel amaçlı sanayi makineleri",
}

def _yuzde_deger(inf: Dict[str, Any], anahtar: str) -> Optional[float]:
    try:
        v = inf.get(anahtar)
        return float(v) * 100 if v is not None else None
    except (TypeError, ValueError):
        return None

def turkce_sirket_ozeti(kod: str, inf: Dict[str, Any]) -> str:
    ad = inf.get("longName") or inf.get("shortName") or kod
    sektor = SEKTOR_TR.get(str(inf.get("sector", "")), inf.get("sector") or "sektör bilgisi bulunmayan")
    endustri = ENDUSTRI_TR.get(str(inf.get("industry", "")), inf.get("industry") or "faaliyet alanı belirtilmeyen")
    ulke = "Türkiye" if str(inf.get("country", "")).lower() in {"turkey", "türkiye"} else (inf.get("country") or "Türkiye")

    cumleler = [f"{ad}, {ulke} merkezli ve ağırlıklı olarak {sektor.lower()} sektöründe faaliyet gösteren bir şirkettir."]
    if endustri:
        cumleler.append(f"Yahoo Finance sınıflandırmasına göre ana faaliyet alanı {str(endustri).lower()} olarak görünmektedir.")

    ciro = _yuzde_deger(inf, "revenueGrowth")
    kar = _yuzde_deger(inf, "earningsGrowth")
    marj = _yuzde_deger(inf, "profitMargins")
    roe = _yuzde_deger(inf, "returnOnEquity")

    finans = []
    if ciro is not None:
        finans.append(f"ciro büyümesi %{ciro:.1f}")
    if kar is not None:
        finans.append(f"kâr büyümesi %{kar:.1f}")
    if marj is not None:
        finans.append(f"net kâr marjı %{marj:.1f}")
    if roe is not None:
        finans.append(f"özsermaye kârlılığı %{roe:.1f}")
    if finans:
        cumleler.append("Mevcut verilerde " + ", ".join(finans) + " seviyesindedir.")
    cumleler.append("Bu özet otomatik üretilmiştir; bilanço ve KAP açıklamalarıyla ayrıca doğrulanmalıdır.")
    return " ".join(cumleler)

# -------------------------
# Dip dönüşü ve işlem planı
# -------------------------
def _pozitif_uyumsuzluk(close: pd.Series, rsi: pd.Series) -> bool:
    if len(close) < 30:
        return False
    ilk = slice(-30, -15)
    ikinci = slice(-15, None)
    fiyat1, fiyat2 = float(close.iloc[ilk].min()), float(close.iloc[ikinci].min())
    rsi1, rsi2 = float(rsi.iloc[ilk].min()), float(rsi.iloc[ikinci].min())
    return fiyat2 < fiyat1 and rsi2 > rsi1 + 2


def islem_plani_hesapla(ticker_name: str, df_raw: Optional[pd.DataFrame] = None) -> Optional[TradePlan]:
    df = fiyat_verisi_getir(ticker_name, "2y") if df_raw is None else df_raw.copy()
    if df.empty or len(df) < 55:
        return None
    x = gostergeleri_hesapla(df)
    son = x.iloc[-1]
    onceki = x.iloc[-2]
    fiyat = float(son["Close"])
    atr = float(son["ATR"]) if pd.notna(son["ATR"]) else fiyat * 0.025
    atr_pct = 100 * atr / fiyat if fiyat > 0 else 0
    rsi = float(son["RSI"])
    gunluk = 100 * (fiyat / float(onceki["Close"]) - 1)

    puan = 0
    nedenler: List[str] = []
    uyarilar: List[str] = []

    # Ucuzluk / dip bölgesi
    if 28 <= rsi <= 42:
        puan += 18
        nedenler.append(f"RSI {rsi:.1f}: dip bölgesine yakın")
    elif 42 < rsi <= 50:
        puan += 10
        nedenler.append(f"RSI {rsi:.1f}: nötr-alt bölge")
    elif rsi < 28:
        puan += 8
        uyarilar.append("RSI aşırı satımda; düşüş henüz bitmemiş olabilir")
    elif rsi >= 70:
        puan -= 15
        uyarilar.append("RSI aşırı alım bölgesinde")

    bb_alt = float(son["BB_LOWER"]) if pd.notna(son["BB_LOWER"]) else fiyat
    bb_orta = float(son["SMA20"]) if pd.notna(son["SMA20"]) else fiyat
    if fiyat <= bb_alt * 1.03:
        puan += 12
        nedenler.append("Fiyat Bollinger alt bandına yakın")
    if fiyat > bb_alt and float(onceki["Close"]) <= float(onceki["BB_LOWER"] if pd.notna(onceki["BB_LOWER"]) else onceki["Close"]):
        puan += 8
        nedenler.append("Bollinger alt bandından yukarı dönüş")

    sma50 = float(son["SMA50"]) if pd.notna(son["SMA50"]) else fiyat
    if abs(fiyat - sma50) / max(sma50, 1e-9) <= 0.04:
        puan += 10
        nedenler.append("50 günlük ortalamaya yakın destek")

    # Dönüş teyitleri
    macd_yukari = son["MACD"] > son["MACD_SIGNAL"] and onceki["MACD"] <= onceki["MACD_SIGNAL"]
    if macd_yukari:
        puan += 14
        nedenler.append("MACD yukarı kesişim verdi")
    elif son["MACD_HIST"] > onceki["MACD_HIST"]:
        puan += 6
        nedenler.append("MACD momentumu iyileşiyor")

    stoch_yukari = son["STOCH_K"] > son["STOCH_D"] and onceki["STOCH_K"] <= onceki["STOCH_D"]
    if stoch_yukari and son["STOCH_K"] < 45:
        puan += 8
        nedenler.append("Stokastik dipten yukarı kesişti")

    vol_ratio = float(son["VOL_RATIO"]) if pd.notna(son["VOL_RATIO"]) else 0
    if vol_ratio >= 1.5 and gunluk > 0:
        puan += 12
        nedenler.append(f"Pozitif günde hacim ortalamanın {vol_ratio:.1f} katı")
    elif vol_ratio >= 1.1:
        puan += 5
        nedenler.append("Hacim ortalamanın üzerinde")

    if _pozitif_uyumsuzluk(x["Close"], x["RSI"]):
        puan += 12
        nedenler.append("RSI pozitif uyumsuzluğu tespit edildi")

    band_width = float(son["BB_WIDTH"]) if pd.notna(son["BB_WIDTH"]) else 1
    if band_width < 0.12:
        puan += 6
        nedenler.append("Bollinger bantları sıkışmış")

    if fiyat > float(son["EMA9"]):
        puan += 5
        nedenler.append("Fiyat EMA9 üzerine çıktı")
    if fiyat > float(son["EMA20"]):
        puan += 5
        nedenler.append("Fiyat EMA20 üzerinde")

    if float(son["ROC20"]) < -18:
        puan -= 8
        uyarilar.append("Son 20 günlük düşüş çok sert")
    if atr_pct > 7:
        puan -= 8
        uyarilar.append("Oynaklık çok yüksek")

    haber_skor, haber_notu = haber_duygu_skori(ticker_name)
    puan += haber_skor * 2
    if haber_skor != 0:
        nedenler.append(f"{haber_notu}: {haber_skor:+d}")

    dip_puani = int(max(0, min(100, puan)))

    # Aşama sınıflandırması
    teyit_sayisi = sum([
        bool(macd_yukari),
        bool(stoch_yukari),
        bool(fiyat > float(son["EMA9"])),
        bool(vol_ratio >= 1.2 and gunluk > 0),
    ])
    if dip_puani >= 75 and teyit_sayisi >= 2:
        asama = "🟢 GÜÇLÜ DİP DÖNÜŞÜ"
    elif dip_puani >= 60 and teyit_sayisi >= 1:
        asama = "🟡 ALIM BÖLGESİ / TEYİTLİ"
    elif dip_puani >= 50:
        asama = "🟠 DİP ADAYI / TEYİT BEKLE"
    else:
        asama = "🔴 UYGUN DEĞİL"

    destekler, direncler = destek_direnc_bul(x)
    alt_destekler = [s for s in destekler if s < fiyat]
    ust_direncler = [r for r in direncler if r > fiyat]
    yakin_destek = max(alt_destekler) if alt_destekler else float(x["Low"].tail(20).min())

    alim_alt = max(yakin_destek, fiyat - 0.55 * atr)
    alim_ust = fiyat + 0.20 * atr

    # Stop: destek altı ve ATR stopundan daha güvenli olanı
    stop_destek = yakin_destek - 0.55 * atr
    stop_atr = fiyat - max(1.6 * atr, fiyat * 0.025)
    stop = min(stop_destek, stop_atr)
    stop = max(stop, fiyat * 0.82)

    risk = max(fiyat - stop, fiyat * 0.01)
    # Direnç + ATR + Bollinger tabanlı hedef birleşimi
    teknik_h1 = fiyat + 2.0 * risk
    teknik_h2 = fiyat + 3.2 * risk
    bb_hedef = max(bb_orta, fiyat + 1.2 * atr)
    direnc1 = next((r for r in sorted(ust_direncler) if r >= fiyat + 1.2 * atr), None)
    direnc2 = next((r for r in sorted(ust_direncler) if direnc1 is not None and r > direnc1 * 1.015), None)

    hedef1_aday = [teknik_h1, bb_hedef]
    if direnc1:
        hedef1_aday.append(direnc1)
    hedef_1 = float(np.median(hedef1_aday))
    hedef_1 = max(hedef_1, fiyat + 1.5 * risk)

    hedef2_aday = [teknik_h2, fiyat + 4 * atr]
    if direnc2:
        hedef2_aday.append(direnc2)
    elif direnc1:
        hedef2_aday.append(direnc1 + 2 * atr)
    hedef_2 = float(np.median(hedef2_aday))
    hedef_2 = max(hedef_2, hedef_1 + 0.8 * atr)

    pot1 = 100 * (hedef_1 / fiyat - 1)
    pot2 = 100 * (hedef_2 / fiyat - 1)
    risk_pct = 100 * (fiyat - stop) / fiyat
    rr = (hedef_1 - fiyat) / max(fiyat - stop, 1e-9)

    # Güven: dip puanı + teyit + kabul edilebilir risk/kazanç
    guven = dip_puani
    guven += 5 if rr >= 2 else -8
    guven += 5 if risk_pct <= 5 else -5
    guven += min(teyit_sayisi * 3, 10)
    sinyal_guveni = int(max(0, min(100, guven)))

    tahmini_sure = "3–15 işlem günü" if atr_pct >= 4 else "5–25 işlem günü"

    return TradePlan(
        ticker=ticker_name,
        fiyat=round(fiyat, 2),
        dip_puani=dip_puani,
        sinyal_guveni=sinyal_guveni,
        asama=asama,
        alim_alt=round(alim_alt, 2),
        alim_ust=round(alim_ust, 2),
        hedef_1=round(hedef_1, 2),
        hedef_2=round(hedef_2, 2),
        stop=round(stop, 2),
        potansiyel_1=round(pot1, 2),
        potansiyel_2=round(pot2, 2),
        risk_yuzde=round(risk_pct, 2),
        risk_kazanc=round(rr, 2),
        atr=round(atr, 2),
        atr_yuzde=round(atr_pct, 2),
        rsi=round(rsi, 1),
        gunluk_degisim=round(gunluk, 2),
        tahmini_sure=tahmini_sure,
        nedenler=nedenler[:8],
        uyarilar=uyarilar[:5],
    )


def dinamik_trailing_stop(plan: TradePlan, pik_fiyat: float, guncel_fiyat: float) -> float:
    # Kâr büyüdükçe stop sıkılaşır; oynaklık yüksekse daha geniş kalır.
    kar_pct = 100 * (guncel_fiyat / max(plan.fiyat, 1e-9) - 1)
    atr_carpani = 2.2
    if kar_pct >= 15:
        atr_carpani = 1.25
    elif kar_pct >= 8:
        atr_carpani = 1.55
    elif kar_pct >= 4:
        atr_carpani = 1.85
    stop = pik_fiyat - atr_carpani * plan.atr
    return round(max(plan.stop, stop), 2)


def sinyal_kaydet(plan: TradePlan) -> None:
    gecmis = veri_yukle(SINYAL_DOSYASI, [])
    bugun = datetime.now().strftime("%Y-%m-%d")
    anahtar = f"{plan.ticker}-{bugun}-{plan.asama}"
    if any(k.get("anahtar") == anahtar for k in gecmis):
        return
    kayit = asdict(plan)
    kayit.update({"anahtar": anahtar, "tarih": datetime.now().isoformat(timespec="minutes")})
    gecmis.append(kayit)
    veri_kaydet(SINYAL_DOSYASI, gecmis[-5000:])


# -------------------------
# Basit yürüyen backtest
# -------------------------
def backtest_yap(ticker_name: str, gun: int = 500, min_puan: int = 65) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = fiyat_verisi_getir(ticker_name, "5y")
    if df.empty or len(df) < 250:
        return pd.DataFrame(), {}

    baslangic = max(220, len(df) - gun)
    islemler: List[Dict[str, Any]] = []
    pozisyon_bitis = -1

    for i in range(baslangic, len(df) - 25):
        if i <= pozisyon_bitis:
            continue
        parcali = df.iloc[: i + 1]
        plan = islem_plani_hesapla(ticker_name, parcali)
        if plan is None or plan.dip_puani < min_puan or "UYGUN DEĞİL" in plan.asama or "TEYİT BEKLE" in plan.asama:
            continue

        giris = float(df["Close"].iloc[i])
        stop = plan.stop
        hedef = plan.hedef_1
        sonuc = "SÜRE DOLDU"
        cikis = float(df["Close"].iloc[min(i + 20, len(df) - 1)])
        cikis_i = min(i + 20, len(df) - 1)

        for j in range(i + 1, min(i + 21, len(df))):
            gun_low = float(df["Low"].iloc[j])
            gun_high = float(df["High"].iloc[j])
            # Aynı gün ikisi de görülürse ihtiyatlı olarak stop önce kabul edilir.
            if gun_low <= stop:
                sonuc, cikis, cikis_i = "STOP", stop, j
                break
            if gun_high >= hedef:
                sonuc, cikis, cikis_i = "HEDEF", hedef, j
                break

        getiri = 100 * (cikis / giris - 1)
        islemler.append({
            "Tarih": df.index[i].strftime("%Y-%m-%d"),
            "Giriş": round(giris, 2),
            "Hedef": round(hedef, 2),
            "Stop": round(stop, 2),
            "Sonuç": sonuc,
            "Çıkış": round(cikis, 2),
            "Getiri %": round(getiri, 2),
            "Dip Puanı": plan.dip_puani,
        })
        pozisyon_bitis = cikis_i

    sonuc_df = pd.DataFrame(islemler)
    if sonuc_df.empty:
        return sonuc_df, {}

    getiriler = sonuc_df["Getiri %"] / 100
    toplam = len(sonuc_df)
    kazanan = int((getiriler > 0).sum())
    brut_kar = getiriler[getiriler > 0].sum()
    brut_zarar = abs(getiriler[getiriler < 0].sum())
    equity = (1 + getiriler).cumprod()
    drawdown = equity / equity.cummax() - 1
    ozet = {
        "Toplam işlem": toplam,
        "Başarı oranı %": round(100 * kazanan / toplam, 1),
        "Ortalama getiri %": round(100 * getiriler.mean(), 2),
        "Bileşik getiri %": round(100 * (equity.iloc[-1] - 1), 2),
        "Kâr faktörü": round(brut_kar / brut_zarar, 2) if brut_zarar > 0 else math.inf,
        "Maksimum düşüş %": round(100 * drawdown.min(), 2),
    }
    return sonuc_df, ozet


# -------------------------
# Alarm kontrolü
# -------------------------
def alarmlari_kontrol_et() -> List[str]:
    alarmlar = veri_yukle(ALARMLAR_DOSYASI, [])
    tetiklenenler: List[str] = []
    degisti = False
    for alarm in alarmlar:
        if alarm.get("durum") != "AKTİF":
            continue
        fiyat = guncel_fiyat_bul(alarm["hisse"])
        if fiyat is None:
            continue
        hedef = float(alarm["fiyat"])
        yon = alarm["yon"]
        tetik = (yon == "GEÇİNCE" and fiyat >= hedef) or (yon == "DÜŞÜNCE" and fiyat <= hedef)
        if tetik:
            alarm["durum"] = "TETİKLENDİ"
            alarm["tetik_tarihi"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            alarm["tetik_fiyati"] = fiyat
            metin = f"🚨 {alarm['hisse']} alarmı tetiklendi: {fiyat:.2f} TL ({yon} {hedef:.2f} TL)"
            tetiklenenler.append(metin)
            telegram_bildirim_gonder(metin)
            degisti = True
    if degisti:
        veri_kaydet(ALARMLAR_DOSYASI, alarmlar)
    return tetiklenenler


# =========================================================
# ARAYÜZ
# =========================================================
if "notified_stocks" not in st.session_state:
    st.session_state.notified_stocks = {}

portfoy: Dict[str, Dict[str, float]] = veri_yukle(PORTFOY_DOSYASI, {})
pik_hafiza: Dict[str, float] = veri_yukle(PIK_DOSYASI, {})

st.sidebar.header("💼 Portföyüm & Takip")
with st.sidebar.expander("➕ Portföye Hisse Ekle"):
    yeni_hisse = st.selectbox("Hisse seç", BIST_OTOMATIK_HAVUZ)
    adet = st.number_input("Adet", min_value=1, step=1, value=100)
    maliyet = st.number_input("Maliyet (TL)", min_value=0.01, step=0.05, value=10.0)
    if st.button("Portföye kaydet", use_container_width=True):
        portfoy[yeni_hisse] = {"adet": int(adet), "maliyet": float(maliyet)}
        veri_kaydet(PORTFOY_DOSYASI, portfoy)
        st.success(f"{yeni_hisse} eklendi")
        st.rerun()

with st.sidebar.expander("🗑️ Portföyden Hisse Sil"):
    if portfoy:
        silinecek = st.selectbox("Silinecek hisse", sorted(portfoy))
        if st.button("Portföyden çıkar", use_container_width=True):
            portfoy.pop(silinecek, None)
            pik_hafiza.pop(silinecek, None)
            veri_kaydet(PORTFOY_DOSYASI, portfoy)
            veri_kaydet(PIK_DOSYASI, pik_hafiza)
            st.rerun()
    else:
        st.info("Portföy boş")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Canlı Portföy")
toplam_deger = toplam_maliyet = 0.0
for h, bilgi in portfoy.items():
    fiyat = guncel_fiyat_bul(h) or float(bilgi["maliyet"])
    adet_b = float(bilgi["adet"])
    maliyet_b = float(bilgi["maliyet"])
    deger = fiyat * adet_b
    maliyet_toplam = maliyet_b * adet_b
    kz = deger - maliyet_toplam
    kz_pct = 100 * kz / maliyet_toplam if maliyet_toplam else 0
    toplam_deger += deger
    toplam_maliyet += maliyet_toplam
    st.sidebar.markdown(f"**{h}** — {fiyat:.2f} TL | {kz_pct:+.2f}%")

if portfoy:
    net = toplam_deger - toplam_maliyet
    st.sidebar.metric("Toplam değer", f"{tr_fiyat(toplam_deger)} TL")
    st.sidebar.metric("Net K/Z", f"{tr_fiyat(net)} TL", f"{100*net/toplam_maliyet:+.2f}%" if toplam_maliyet else "0%")

st.title("🛡️ Canavar AI Trade Terminal v4.0")
col_sub1, col_sub2 = st.columns([0.82, 0.18])
with col_sub1:
    st.caption("Dip dönüşü • hedef fiyat • ATR tabanlı dinamik stop • alarm • backtest")
with col_sub2:
    reset_key = datetime.now().strftime("%Y%m%d%H%M%S")
    components.html(
        f"""
        <div style="font-family:Arial;display:flex;justify-content:flex-end;gap:6px;align-items:center">
          <span style="color:#888;font-size:11px">Yenileme:</span>
          <span id="countdown_{reset_key}" style="font-size:13px;font-weight:bold;background:#1f232a;padding:2px 8px;border-radius:4px">60s</span>
        </div>
        <script>
          let t=60; const e=document.getElementById('countdown_{reset_key}');
          const id=setInterval(()=>{{if(t<=0){{e.innerHTML='Yenileniyor...';clearInterval(id)}}else{{e.innerHTML=t+'s';t--;}}}},1000);
        </script>
        """,
        height=32,
    )

st.warning(
    "Bu yazılım yatırım danışmanlığı değildir. Dip, hedef ve olasılık değerleri geçmiş fiyat/hacim verilerinden üretilen karar destek tahminleridir; kesin sonuç vermez."
)

for mesaj in alarmlari_kontrol_et():
    st.error(mesaj)

t1, t2, t3, t4, t5 = st.tabs([
    "🎯 Dipten Al / Pikten Sat",
    "💼 Portföy Koruma",
    "🧪 Backtest",
    "📖 Temel Analiz",
    "🚨 Alarmlar",
])

with t1:
    st.header("🎯 BIST Dip Dönüşü Tarayıcısı")
    c1, c2, c3 = st.columns(3)
    with c1:
        minimum_puan = st.slider("Minimum dip puanı", 40, 90, 60)
    with c2:
        sadece_teyitli = st.checkbox(
            "Yalnızca teyitli sinyaller",
            value=False,
            help="Açılırsa yalnızca dönüş teyidi bulunan hisseler gösterilir; sonuç sayısı ciddi şekilde azalabilir.",
        )
    with c3:
        maksimum_hisse = st.number_input("En fazla sonuç", 5, 50, 20)

    if st.button("🔎 Tüm havuzu tara", type="primary", use_container_width=True):
        bar = st.progress(0)
        durum = st.empty()
        uygun_sonuclar: List[Dict[str, Any]] = []
        tum_adaylar: List[Dict[str, Any]] = []
        planlar: Dict[str, TradePlan] = {}
        veri_alinamayan: List[str] = []
        puan_altinda = 0
        teyitsiz = 0

        for i, h in enumerate(BIST_OTOMATIK_HAVUZ):
            bar.progress((i + 1) / len(BIST_OTOMATIK_HAVUZ))
            durum.caption(f"Taranıyor: {h} ({i + 1}/{len(BIST_OTOMATIK_HAVUZ)})")
            plan = islem_plani_hesapla(h)
            if plan is None:
                veri_alinamayan.append(h)
                continue

            planlar[h] = plan
            satir = {
                "Hisse": h,
                "Fiyat": f"{plan.fiyat:.2f}",
                "Dip Puanı": plan.dip_puani,
                "Güven": plan.sinyal_guveni,
                "Aşama": plan.asama,
                "Alım Bölgesi": f"{plan.alim_alt:.2f}–{plan.alim_ust:.2f}",
                "Hedef 1": f"{plan.hedef_1:.2f} (%{plan.potansiyel_1:.1f})",
                "Hedef 2": f"{plan.hedef_2:.2f} (%{plan.potansiyel_2:.1f})",
                "Stop": f"{plan.stop:.2f} (-%{plan.risk_yuzde:.1f})",
                "R/K": plan.risk_kazanc,
                "RSI": plan.rsi,
            }
            tum_adaylar.append(satir)

            if plan.dip_puani < minimum_puan:
                puan_altinda += 1
                continue
            if sadece_teyitli and ("TEYİT BEKLE" in plan.asama or "UYGUN DEĞİL" in plan.asama):
                teyitsiz += 1
                continue
            sinyal_kaydet(plan)
            uygun_sonuclar.append(satir)

        bar.empty()
        durum.empty()
        siralama = lambda z: (z["Güven"], z["Dip Puanı"], z["R/K"])
        uygun_sonuclar = sorted(uygun_sonuclar, key=siralama, reverse=True)[: int(maksimum_hisse)]
        tum_adaylar = sorted(tum_adaylar, key=siralama, reverse=True)

        esik_esnetildi = False
        if not uygun_sonuclar and tum_adaylar:
            # Kullanıcı boş ekran görmesin: en yüksek puanlıları izleme listesi olarak göster.
            uygun_sonuclar = tum_adaylar[: min(int(maksimum_hisse), 10)]
            esik_esnetildi = True

        st.session_state["tarama_sonuclari"] = uygun_sonuclar
        st.session_state["tarama_planlari"] = {k: asdict(v) for k, v in planlar.items()}
        st.session_state["tarama_ozeti"] = {
            "veri_alinan": len(tum_adaylar),
            "veri_alinamayan": veri_alinamayan,
            "puan_altinda": puan_altinda,
            "teyitsiz": teyitsiz,
            "esik_esnetildi": esik_esnetildi,
            "minimum_puan": minimum_puan,
        }

    sonuclar = st.session_state.get("tarama_sonuclari", [])
    tarama_ozeti = st.session_state.get("tarama_ozeti", {})
    if sonuclar:
        if tarama_ozeti.get("esik_esnetildi"):
            st.warning(
                f"Seçtiğiniz koşullara tam uyan hisse bulunamadı. Bunun yerine en yüksek puanlı "
                f"{len(sonuclar)} hisse izleme adayı olarak gösteriliyor. Bunlar doğrudan al sinyali değildir."
            )
        else:
            st.success(f"{len(sonuclar)} aday listelendi")
        if tarama_ozeti:
            st.caption(
                f"Veri alınan: {tarama_ozeti.get('veri_alinan', 0)} | "
                f"Puan altında kalan: {tarama_ozeti.get('puan_altinda', 0)} | "
                f"Teyit filtresinde elenen: {tarama_ozeti.get('teyitsiz', 0)} | "
                f"Veri alınamayan: {len(tarama_ozeti.get('veri_alinamayan', []))}"
            )
            if tarama_ozeti.get("veri_alinamayan"):
                with st.expander("Verisi alınamayan hisseleri göster"):
                    st.write(", ".join(tarama_ozeti["veri_alinamayan"]))
        st.dataframe(pd.DataFrame(sonuclar), use_container_width=True, hide_index=True)
        secili = st.selectbox("Detayını göster", [s["Hisse"] for s in sonuclar])
        plan_dict = st.session_state.get("tarama_planlari", {}).get(secili)
        if plan_dict:
            p = TradePlan(**plan_dict)
            st.subheader(f"{p.ticker} işlem planı")
            a, b, c, d = st.columns(4)
            a.metric("Dip puanı", f"{p.dip_puani}/100")
            b.metric("Sinyal güveni", f"{p.sinyal_guveni}/100")
            c.metric("Hedef 1 potansiyeli", f"%{p.potansiyel_1:.1f}")
            d.metric("Risk/Kazanç", f"1:{p.risk_kazanc:.2f}")
            st.info(
                f"**{p.asama}**  |  Alım: **{p.alim_alt:.2f}–{p.alim_ust:.2f} TL**  |  "
                f"Hedef 1: **{p.hedef_1:.2f} TL**  |  Hedef 2: **{p.hedef_2:.2f} TL**  |  "
                f"Stop: **{p.stop:.2f} TL**  |  Tahmini süre: **{p.tahmini_sure}**"
            )
            st.markdown("**Neden seçildi?**")
            for n in p.nedenler:
                st.write(f"• {n}")
            if p.uyarilar:
                st.markdown("**Risk uyarıları**")
                for u in p.uyarilar:
                    st.write(f"• {u}")
    else:
        st.info("Tarama butonuna basın. İlk tarama veri indirdiği için biraz ağır çalışabilir.")

with t2:
    st.header("💼 Portföyde Pikten Satış ve Kâr Koruma")
    if not portfoy:
        st.info("Önce yan menüden portföye hisse ekleyin.")
    else:
        sat_uyarilari = []
        satirlar = []
        for h, bilgi in portfoy.items():
            plan = islem_plani_hesapla(h)
            fiyat = guncel_fiyat_bul(h)
            if plan is None or fiyat is None:
                continue
            eski_pik = float(pik_hafiza.get(h, fiyat))
            pik = max(eski_pik, fiyat)
            pik_hafiza[h] = pik
            dinamik_stop = dinamik_trailing_stop(plan, pik, fiyat)
            pikten_dusus = 100 * (pik - fiyat) / pik if pik else 0
            hedef1_gecti = pik >= plan.hedef_1
            hedef2_gecti = pik >= plan.hedef_2
            if fiyat <= dinamik_stop:
                sinyal = "🚨 SAT / KÂRI KORU"
                sat_uyarilari.append(f"{h}: {fiyat:.2f} TL, dinamik stop {dinamik_stop:.2f} TL")
                anahtar = f"{h}-{dinamik_stop:.2f}"
                if not st.session_state.notified_stocks.get(anahtar):
                    telegram_bildirim_gonder(
                        f"🚨 {h} KÂR KORUMA UYARISI\nGüncel: {fiyat:.2f} TL\nPik: {pik:.2f} TL\nDinamik stop: {dinamik_stop:.2f} TL\nPikten düşüş: %{pikten_dusus:.2f}"
                    )
                    st.session_state.notified_stocks[anahtar] = True
            elif hedef2_gecti:
                sinyal = "🟣 HEDEF 2 GÖRÜLDÜ / STOP SIKILAŞTIR"
            elif hedef1_gecti:
                sinyal = "🟢 HEDEF 1 GÖRÜLDÜ / KISMİ KÂR"
            else:
                sinyal = "🟡 POZİSYONU İZLE"
            satirlar.append({
                "Hisse": h,
                "Güncel": f"{fiyat:.2f}",
                "Maliyet": f"{float(bilgi['maliyet']):.2f}",
                "Pik": f"{pik:.2f}",
                "Pikten Düşüş": f"%{pikten_dusus:.2f}",
                "Hedef 1": f"{plan.hedef_1:.2f}",
                "Hedef 2": f"{plan.hedef_2:.2f}",
                "Dinamik Stop": f"{dinamik_stop:.2f}",
                "Sinyal": sinyal,
            })
        veri_kaydet(PIK_DOSYASI, pik_hafiza)
        for u in sat_uyarilari:
            st.error(u)
        if satirlar:
            st.dataframe(pd.DataFrame(satirlar), use_container_width=True, hide_index=True)
        if st.button("Portföy piklerini güncel fiyata sıfırla"):
            yeni_pikler = {h: (guncel_fiyat_bul(h) or float(portfoy[h]["maliyet"])) for h in portfoy}
            veri_kaydet(PIK_DOSYASI, yeni_pikler)
            st.success("Pikler sıfırlandı")
            st.rerun()

with t3:
    st.header("🧪 Geçmiş Veri Backtest")
    st.caption("Aynı gün hedef ve stop birlikte görülürse ihtiyatlı şekilde stop önce kabul edilir.")
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        bt_hisse = st.selectbox("Backtest hissesi", BIST_OTOMATIK_HAVUZ, key="bt_hisse")
    with bc2:
        bt_gun = st.selectbox("Test dönemi", [250, 500, 750], index=1)
    with bc3:
        bt_puan = st.slider("Minimum sinyal puanı", 50, 85, 65, key="bt_puan")
    if st.button("Backtest çalıştır", type="primary"):
        with st.spinner("Geçmiş sinyaller test ediliyor..."):
            bt_df, ozet = backtest_yap(bt_hisse, int(bt_gun), int(bt_puan))
        if bt_df.empty:
            st.warning("Bu koşullarda yeterli işlem bulunamadı.")
        else:
            cols = st.columns(len(ozet))
            for col, (k, v) in zip(cols, ozet.items()):
                col.metric(k, v)
            st.dataframe(bt_df.sort_values("Tarih", ascending=False), use_container_width=True, hide_index=True)
            getiri_serisi = (1 + bt_df["Getiri %"] / 100).cumprod()
            st.line_chart(pd.DataFrame({"Sermaye çarpanı": getiri_serisi.values}, index=pd.to_datetime(bt_df["Tarih"])))

with t4:
    st.header("📖 Şirket Temel Analiz Defteri")
    secilen_temel = st.selectbox("Şirket", BIST_OTOMATIK_HAVUZ, key="temel_hisse")
    if st.button("Temel verileri çek"):
        inf = temel_veri_getir(secilen_temel)
        if not inf:
            st.error("Temel veri alınamadı")
        else:
            st.write(f"### {secilen_temel}")
            st.info(turkce_sirket_ozeti(secilen_temel, inf))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("F/K", round(inf.get("trailingPE"), 2) if inf.get("trailingPE") else "N/A")
            c2.metric("PD/DD", round(inf.get("priceToBook"), 2) if inf.get("priceToBook") else "N/A")
            c3.metric("ROE", f"%{100*inf.get('returnOnEquity'):.1f}" if inf.get("returnOnEquity") else "N/A")
            c4.metric("Borç/Özsermaye", round(inf.get("debtToEquity"), 1) if inf.get("debtToEquity") else "N/A")
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Net kâr marjı", f"%{100*inf.get('profitMargins'):.1f}" if inf.get("profitMargins") else "N/A")
            c6.metric("Ciro büyümesi", f"%{100*inf.get('revenueGrowth'):.1f}" if inf.get("revenueGrowth") else "N/A")
            c7.metric("Kâr büyümesi", f"%{100*inf.get('earningsGrowth'):.1f}" if inf.get("earningsGrowth") else "N/A")
            c8.metric("Temettü verimi", f"%{100*inf.get('dividendYield'):.2f}" if inf.get("dividendYield") else "Yok/N/A")

with t5:
    st.header("🚨 Akıllı Fiyat Alarmları")
    alarmlar = veri_yukle(ALARMLAR_DOSYASI, [])
    a1, a2, a3 = st.columns(3)
    with a1:
        a_hisse = st.selectbox("Alarm hissesi", BIST_OTOMATIK_HAVUZ, key="alarm_h")
    with a2:
        a_fiyat = st.number_input("Hedef fiyat (TL)", min_value=0.01, step=0.05, value=20.0)
    with a3:
        a_yon = st.selectbox("Yön", ["GEÇİNCE", "DÜŞÜNCE"])
    if st.button("Alarmı kur"):
        alarmlar.append({
            "hisse": a_hisse,
            "fiyat": float(a_fiyat),
            "yon": a_yon,
            "durum": "AKTİF",
            "tarih": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        veri_kaydet(ALARMLAR_DOSYASI, alarmlar)
        st.success("Alarm kaydedildi")
        st.rerun()
    if alarmlar:
        st.dataframe(pd.DataFrame(alarmlar), use_container_width=True, hide_index=True)
        if st.button("Tüm alarmları temizle"):
            veri_kaydet(ALARMLAR_DOSYASI, [])
            st.rerun()
    else:
        st.info("Kurulu alarm yok")

st.markdown("---")
st.caption(
    "Not: Hedefler garanti değildir. Gerçek işlem öncesinde komisyon, kayma, likidite, devre kesici ve veri gecikmesini ayrıca dikkate alın."
)

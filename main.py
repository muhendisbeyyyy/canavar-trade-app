from datetime import datetime
import json
import os
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
import yfinance as yf

# ==========================================
# ⚡ KESİNTİSİZ AUTO-REFRESH (1 DAKİKA = 60.000 MS)
# ==========================================
st_autorefresh(interval=60000, key="canavar_autorefresh")

st.set_page_config(
    page_title="Canavar AI Trade Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 📲 TELEGRAM BİLDİRİM FONKSİYONU
# ==========================================
TELEGRAM_TOKEN = "8887451053:AAHszl4Q53MGxdv5cXETLEuE4IHDxq3jgEo"
TELEGRAM_CHAT_ID = "BURAYA_CHAT_ID_YAZ"  # Kendi Telegram Chat ID'nizi girin


def telegram_bildirim_gonder(mesaj):
    if TELEGRAM_CHAT_ID == "BURAYA_CHAT_ID_YAZ":
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": mesaj,
        "parse_mode": "Markdown",
    }
    try:
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        print(f"Telegram gönderme hatası: {e}")


# ==========================================
# 🎯 PİK FİYAT & BİLDİRİM KALICI HAFIZASI (JSON)
# ==========================================
PIK_DOSYASI = "pik_fiyatlar.json"
PORTFOY_DOSYASI = "portfoy_data.json"
ALARMLAR_DOSYASI = "alarmlar_data.json"

BIST_OTOMATIK_HAVUZ = [
    "AGHOL",
    "AKBNK",
    "AKCNS",
    "AKSEN",
    "ALARK",
    "ALBRK",
    "ALFAS",
    "ARCLK",
    "ASELS",
    "ASTOR",
    "BIMAS",
    "BRSAN",
    "BRYAT",
    "BUCIM",
    "CCOLA",
    "CIMSA",
    "CWENE",
    "DOAS",
    "DOHOL",
    "ECILC",
    "EGEEN",
    "EKGYO",
    "ENJSA",
    "ENKAI",
    "EREGL",
    "EUPWR",
    "FROTO",
    "GARAN",
    "GESAN",
    "GUBRF",
    "HALKB",
    "HEKTS",
    "IMASM",
    "IPEKE",
    "ISCTR",
    "ISGYO",
    "ISMEN",
    "IZMDC",
    "KARDMD",
    "KCHOL",
    "KONTR",
    "KONYA",
    "KOZAA",
    "KOZAL",
    "KCAER",
    "MAVI",
    "MGROS",
    "MIATK",
    "ODAS",
    "OTKAR",
    "OYAKC",
    "PETKM",
    "PGSUS",
    "QUAGR",
    "REEDR",
    "SAHOL",
    "SASA",
    "SAYAS",
    "SISE",
    "SKBNK",
    "SMRTG",
    "SOKM",
    "TABGD",
    "TARKM",
    "TCELL",
    "THYAO",
    "TKFEN",
    "TOASO",
    "TSKB",
    "TTKOM",
    "TUPRS",
    "TURSG",
    "ULKER",
    "VAKBN",
    "VESTL",
    "YEOTK",
    "YKBNK",
    "ZOREN",
]


def veri_yukle(dosya_adi, varsayilan):
    if os.path.exists(dosya_adi):
        try:
            with open(dosya_adi, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return varsayilan
    return varsayilan


def veri_kaydet(dosya_adi, veri):
    with open(dosya_adi, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False, indent=4)


if "notified_stocks" not in st.session_state:
    st.session_state.notified_stocks = {}


# 🛡️ GÜVENLİ FİYAT ÇEKME FONKSİYONU
def guncel_fiyat_bul(ticker_name):
    try:
        ticker = yf.Ticker(f"{ticker_name}.IS")
        df = ticker.history(period="5d")
        if not df.empty and "Close" in df.columns:
            gecerli_fiyatlar = df["Close"].dropna()
            if not gecerli_fiyatlar.empty:
                return round(float(gecerli_fiyatlar.iloc[-1]), 2)
        return None
    except:
        return None


def yapay_zeka_haber_analizi(ticker_name):
    try:
        ticker = yf.Ticker(f"{ticker_name}.IS")
        haberler = ticker.news
        if not haberler:
            return 0
        pozitif = [
            "halka arz",
            "kazanc",
            "kar ",
            "buyume",
            "ortaklik",
            "sozlesme",
            "ihale",
            "rekor",
            "alim",
            "pozitif",
            "yukselis",
            "yeni is",
            "kap",
            "temettu",
            "bedelsiz",
            "yatirim",
        ]
        negatif = [
            "zarar",
            "dusus",
            "kayip",
            "dava",
            "ceza",
            "iptal",
            "borc",
            "temerrut",
            "negatif",
            "satis",
            "risk",
        ]
        skor = 0
        for h in haberler[:3]:
            baslik = h.get("title", "").lower()
            for p in pozitif:
                if p in baslik:
                    skor += 1
            for n in negatif:
                if n in baslik:
                    skor -= 1
        return 2 if skor >= 2 else (-2 if skor <= -2 else 0)
    except:
        return 0


def canavar_teknik_analiz(ticker_name):
    try:
        ticker = yf.Ticker(f"{ticker_name}.IS")
        df = ticker.history(period="100d", interval="1d")
        if len(df) < 50:
            return 0, False, 0.0, 50.0

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]
        onay = 0

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1])
        if 35 < current_rsi < 65:
            onay += 1

        # MACD
        macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        signal = macd.ewm(span=9).mean()
        if macd.iloc[-1] > signal.iloc[-1]:
            onay += 1

        # Bollinger
        if close.iloc[-1] > close.rolling(20).mean().iloc[-1]:
            onay += 1

        # Stokastik
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        k = 100 * ((close - low14) / (high14 - low14 + 1e-10))
        d = k.rolling(3).mean()
        if k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 80:
            onay += 1

        # Ortalamalar
        if close.iloc[-1] > close.rolling(50).mean().iloc[-1]:
            onay += 1
        if close.iloc[-1] > close.ewm(9).mean().iloc[-1]:
            onay += 1
        if (
            volume.iloc[-1] > volume.rolling(10).mean().iloc[-1]
            and close.iloc[-1] > close.iloc[-2]
        ):
            onay += 1
        if close.iloc[-1] > close.rolling(10).mean().iloc[-1]:
            onay += 1

        # Günlük Değişim Yüzdesi Hesaplama
        gunluk_degisim = 0.0
        if len(close) >= 2:
            onceki_kapanis = float(close.iloc[-2])
            guncel = float(close.iloc[-1])
            if onceki_kapanis > 0:
                gunluk_degisim = ((guncel - onceki_kapanis) / onceki_kapanis) * 100

        # TEPEDEN ALMA / AŞIRI ISINMA KONTROLÜ
        asiri_isinma = gunluk_degisim >= 3.5 or current_rsi >= 65.0

        return onay, asiri_isinma, gunluk_degisim, current_rsi
    except:
        return 0, False, 0.0, 50.0


# --- PORTFÖY YÖNETİMİ SIDEBAR ---
st.sidebar.header("💼 Portföyüm & Takip")

# ⏱️ DİNAMİK VE HER RERUN'DA SIFIRLANAN YENİLEME SAYACI
st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ Canlı Yenileme Sayacı")

reset_key = datetime.now().strftime("%Y%m%d%H%M%S")
components.html(
    f"""
    <div style="font-family: Arial, sans-serif; text-align: center; padding: 10px; background-color: #1a1c23; border-radius: 8px; border: 1px solid #333;">
        <span style="color: #aaa; font-size: 13px;">Sonraki Yenileme:</span>
        <div id="countdown" style="font-size: 24px; font-weight: bold; color: #00e676; margin-top: 2px;">60 sn</div>
    </div>
    <script>
        var timeLeft = 60;
        var elem = document.getElementById('countdown');
        var timerId = setInterval(countdown, 1000);
        function countdown() {{
            if (timeLeft <= 0) {{
                elem.innerHTML = "Yenileniyor...";
                elem.style.color = "#ff5252";
                clearInterval(timerId);
            }} else {{
                elem.innerHTML = timeLeft + ' sn';
                timeLeft--;
            }}
        }}
    </script>
    """,
    height=80,
    key=f"timer_{reset_key}",
)

portfoy = veri_yukle(PORTFOY_DOSYASI, {})
pik_hafiza = veri_yukle(PIK_DOSYASI, {})

with st.sidebar.expander("➕ Portföye Hisse Ekle"):
    yeni_hisse = st.selectbox("Hisse Seç", BIST_OTOMATIK_HAVUZ)
    adet = st.number_input("Adet", min_value=1, step=1, value=100)
    maliyet = st.number_input(
        "Maliyet (TL)", min_value=0.1, step=0.1, value=10.0
    )
    if st.button("Kaydet"):
        portfoy[yeni_hisse] = {"adet": adet, "maliyet": maliyet}
        veri_kaydet(PORTFOY_DOSYASI, portfoy)
        st.success(f"{yeni_hisse} portföye eklendi!")
        st.rerun()

with st.sidebar.expander("🗑️ Portföyden Hisse Sil"):
    if portfoy:
        silinecek = st.selectbox("Silinecek Hisse", list(portfoy.keys()))
        if st.button("Portföyden Çıkar"):
            del portfoy[silinecek]
            if silinecek in pik_hafiza:
                del pik_hafiza[silinecek]
                veri_kaydet(PIK_DOSYASI, pik_hafiza)
            veri_kaydet(PORTFOY_DOSYASI, portfoy)
            st.warning(f"{silinecek} silindi.")
            st.rerun()
    else:
        st.write("Silinecek hisse yok.")

# --- CANLI PORTFÖY ANALİZİ ---
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Canlı Portföy Durumu")

toplam_deger = 0.0
toplam_maliyet_genel = 0.0

if portfoy:
    for h, bilgi in list(portfoy.items()):
        fiyat = guncel_fiyat_bul(h)
        maliyet_b = float(bilgi["maliyet"])
        adet_b = float(bilgi["adet"])

        if fiyat is None:
            fiyat = maliyet_b

        anlik_deger = fiyat * adet_b
        toplam_maliyet = maliyet_b * adet_b

        toplam_deger += anlik_deger
        toplam_maliyet_genel += toplam_maliyet

        kz_tl = anlik_deger - toplam_maliyet
        kz_yuzde = (
            (kz_tl / toplam_maliyet) * 100 if toplam_maliyet > 0 else 0.0
        )

        renk = "green" if kz_tl >= 0 else "red"
        ok = "🔺" if kz_tl >= 0 else "🔻"

        st.sidebar.markdown(
            f"**{h}** | <span style='color:{renk}'>{ok} {kz_yuzde:+.2f}%</span>",
            unsafe_allow_html=True,
        )
        st.sidebar.write(
            f"Fiyat: {fiyat:.2f} TL | Maliyet: {maliyet_b:.2f} TL"
        )
        st.sidebar.write(f"Net K/Z: {kz_tl:+.2f} TL")
        st.sidebar.write("---")

    net_kz_genel = toplam_deger - toplam_maliyet_genel
    yuzde_kz_genel = (
        (net_kz_genel / toplam_maliyet_genel) * 100
        if toplam_maliyet_genel > 0
        else 0.0
    )

    st.sidebar.subheader("🏆 Toplam Portföy Özeti")
    st.sidebar.metric("Toplam Değer", f"{toplam_deger:,.2f} TL")
    st.sidebar.metric(
        "Toplam Net Kâr/Zarar", f"{net_kz_genel:,.2f} TL", f"{yuzde_kz_genel:+.2f}%"
    )
else:
    st.sidebar.write("Portföyünüz henüz boş.")

# --- ANA EKRAN ---
st.title("🛡️ Canavar AI Trade Terminal v3.3")
st.caption(
    "⚡ Otomatik Yenileme Aktif (1 Dk) | 🎯 Pik Seviye & Trailing Stop Koruma"
    " | 🛑 Tepeden Alma Filtresi"
)

t1, t2, t3 = st.tabs(
    ["🚀 Canavar AI Hibrit Süzgeç", "📖 Temel Analiz Defteri", "🚨 Akıllı Alarmlar"]
)

with t1:
    st.header("📯 Portföydeki Hisselerin Canavar Yapay Zeka Raporu")

    portfoy_tablosu = []
    pik_sat_uyarilari = []

    for h in portfoy.keys():
        fiyat = guncel_fiyat_bul(h)
        tech, asiri_isinma, degisim, current_rsi = canavar_teknik_analiz(h)
        ai = yapay_zeka_haber_analizi(h)
        bilesik = tech + ai

        fiyat_b = fiyat if fiyat is not None else float(portfoy[h]["maliyet"])
        fiyat_str = f"{fiyat_b:.2f}"

        # 🎯 KALICI PİK FİYAT GÜNCELLEME (JSON)
        mevcut_pik = pik_hafiza.get(h, fiyat_b)
        if fiyat_b > mevcut_pik or h not in pik_hafiza:
            mevcut_pik = fiyat_b
            pik_hafiza[h] = mevcut_pik
            veri_kaydet(PIK_DOSYASI, pik_hafiza)
            st.session_state.notified_stocks[h] = False

        zirve_fiyat = pik_hafiza.get(h, fiyat_b)
        dusus_yuzdesi = (
            ((zirve_fiyat - fiyat_b) / zirve_fiyat) * 100
            if zirve_fiyat > 0
            else 0.0
        )

        # 🚨 SERT SİNYAL MANTIĞI: DÜŞÜŞ %1.5 VE ÜZERİYSE DOĞRUDAN SAT!
        if dusus_yuzdesi >= 1.5:
            sinyal = "🚨 SAT / PİK DÖNÜŞÜ"
            mesaj_metni = f"🚨 **CANAVAR AI PİK SAT UYARISI!**\n\n📌 **Hisse:** {h}\n💰 **Güncel Fiyat:** {fiyat_b:.2f} TL\n🔝 **Gördüğü Zirve:** {zirve_fiyat:.2f} TL\n📉 **Zirveden Düşüş:** %{dusus_yuzdesi:.2f}\n📊 **Birleşik Skor:** {bilesik}\n\n⚠️ Kârı korumak için satışı değerlendir!"

            pik_sat_uyarilari.append(
                f"⚠️ **{h}** zirveden (%{dusus_yuzdesi:.2f}) düştü! (Pik:"
                f" {zirve_fiyat:.2f} TL ➔ Güncel: {fiyat_b:.2f} TL)"
            )

            if not st.session_state.notified_stocks.get(h, False):
                telegram_bildirim_gonder(mesaj_metni)
                st.session_state.notified_stocks[h] = True

        elif bilesik >= 7:
            sinyal = "🟢 CANAVAR AL"
        elif bilesik >= 4:
            sinyal = "🟡 NÖTR / BEKLE"
        else:
            sinyal = "🔴 RİSKLİ / SAT"

        ai_aciklama = (
            "🟢 Pozitif Akış"
            if ai > 0
            else ("🔴 Negatif Akış" if ai < 0 else "🟡 Nötr Akış (Aktif Haber Yok)")
        )

        portfoy_tablosu.append({
            "Hisse": h,
            "Fiyat (TL)": fiyat_str,
            "Pik Fiyat (TL)": f"{zirve_fiyat:.2f}",
            "Pikten Düşüş": f"%{dusus_yuzdesi:.2f}",
            "Teknik Skor (Max 8)": f"{tech} Onay",
            "AI Haber Katkısı": f"{ai:+.0f} Pts",
            "Canlı AI Haber/KAP Durumu": ai_aciklama,
            "BİRLEŞİK GÜÇ SKORU": bilesik,
            "İŞLEM SİNYALİ": sinyal,
        })

    if pik_sat_uyarilari:
        for uyari in pik_sat_uyarilari:
            st.error(uyari)

    if portfoy_tablosu:
        st.table(pd.DataFrame(portfoy_tablosu))
    else:
        st.info("Portföyünüzde hisse bulunmadığı için analiz yapılamadı.")

    st.markdown("---")
    st.header(
        "🔥 BIST 75 Havuzunda Yapay Zeka ve Teknik Onaylı Ralli Fırsatları"
    )

    if st.button("🔥 75 HİSSEYİ YAPAY ZEKAYLA BİRLİKTE TARA"):
        tarama_bar = st.progress(0)
        firsatlar = []

        for index, h in enumerate(BIST_OTOMATIK_HAVUZ):
            tarama_bar.progress((index + 1) / len(BIST_OTOMATIK_HAVUZ))
            if h in portfoy:
                continue

            fiyat = guncel_fiyat_bul(h)
            if not fiyat:
                continue

            tech, asiri_isinma, degisim, current_rsi = canavar_teknik_analiz(h)
            if tech >= 4:
                ai = yapay_zeka_haber_analizi(h)
                bilesik = tech + ai

                # 🛑 TEPEDEN ALMA / AŞIRI ISINMA ENGELLEYİCİ MANTIK
                if asiri_isinma:
                    durum_notu = f"🔥 AŞIRI ISINDI (Günlük: %{degisim:+.2f} | RSI: {current_rsi:.1f})"
                elif bilesik >= 7:
                    durum_notu = "🟢 CANAVAR RALLİ FIRSATI"
                else:
                    durum_notu = "🟡 NÖTR / TAKİP"

                if bilesik >= 7:
                    firsatlar.append({
                        "Hisse": h,
                        "Güncel Fiyat": f"{fiyat:.2f} TL",
                        "Günlük Değişim": f"%{degisim:+.2f}",
                        "RSI Değeri": f"{current_rsi:.1f}",
                        "Teknik Onay": f"{tech}/8",
                        "AI Haber Gücü": f"{ai:+.0f}",
                        "Toplam Skor": bilesik,
                        "ALIM DURUMU": durum_notu,
                    })

        tarama_bar.empty()
        if firsatlar:
            st.success(
                f"🔥 Yapay Zeka ve Teknik Filtreleri Aşan {len(firsatlar)} Hisse"
                " Bulundu!"
            )
            df_firsat = pd.DataFrame(firsatlar)
            # Aşırı ısınan tepe hisselerini tablonun altına itin
            df_firsat["Isinma_Sira"] = df_firsat["ALIM DURUMU"].apply(
                lambda x: 1 if "🔥 AŞIRI ISINDI" in x else 0
            )
            df_firsat = df_firsat.sort_values(
                by=["Isinma_Sira", "Toplam Skor"], ascending=[True, False]
            ).drop(columns=["Isinma_Sira"])

            st.dataframe(df_firsat)
        else:
            st.warning("Kriterlere uyan yeni bir ralli fırsatı tespit edilemedi.")

with t2:
    st.subheader("📖 Şirket Temel Analiz Defteri")
    secilen_temel = st.selectbox("Analiz Edilecek Şirket", BIST_OTOMATIK_HAVUZ)
    if st.button("Temel Verileri Çek"):
        try:
            t_obj = yf.Ticker(f"{secilen_temel}.IS")
            inf = t_obj.info
            st.write(f"### {secilen_temel} Genel Bilgiler")
            st.write(inf.get("longBusinessSummary", "Özet bilgi yok."))

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Fiyat/Kazanç (F/K)",
                    round(inf.get("trailingPE", 0), 2)
                    if inf.get("trailingPE")
                    else "N/A",
                )
                st.metric(
                    "Piyasa Değeri / Defter Değeri (PD/DD)",
                    round(inf.get("priceToBook", 0), 2)
                    if inf.get("priceToBook")
                    else "N/A",
                )
            with col2:
                st.metric(
                    "Hisse Başına Kazanç (EPS)",
                    round(inf.get("trailingEps", 0), 2)
                    if inf.get("trailingEps")
                    else "N/A",
                )
                st.metric(
                    "Temettü Verimi (%)",
                    f"{round(inf.get('dividendYield', 0)*100, 2)}%"
                    if inf.get("dividendYield")
                    else "Ödemiyor",
                )
        except:
            st.error("Temel analiz verileri çekilirken bir hata oluştu.")

with t3:
    st.subheader("🚨 Akıllı Fiyat Alarmları")
    alarmlar = veri_yukle(ALARMLAR_DOSYASI, [])

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        a_hisse = st.selectbox("Alarm Hissesi", BIST_OTOMATIK_HAVUZ, key="alarm_h")
    with col_a2:
        a_fiyat = st.number_input(
            "Hedef Fiyat (TL)", min_value=0.1, step=0.05, value=20.0
        )
    with col_a3:
        a_yon = st.selectbox("Yön", ["GEÇİNCE", "DÜŞÜNCE"])

    if st.button("Alarmı Kur"):
        alarmlar.append({
            "hisse": a_hisse,
            "fiyat": a_fiyat,
            "yon": a_yon,
            "durum": "AKTİF",
            "tarih": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        veri_kaydet(ALARMLAR_DOSYASI, alarmlar)
        st.success("Alarm başarıyla kaydedildi!")
        st.rerun()

    if alarmlar:
        st.write("### Mevcut Alarmlarınız")
        df_alarm = pd.DataFrame(alarmlar)
        st.table(df_alarm)
        if st.button("Tüm Alarmları Temizle"):
            veri_kaydet(ALARMLAR_DOSYASI, [])
            st.warning("Bütün alarmlar sıfırlandı.")
            st.rerun()

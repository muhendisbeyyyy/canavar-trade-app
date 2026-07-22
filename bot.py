from datetime import datetime
import json
import os
import threading
import time
import pandas as pd
import telebot
import yfinance as yf

# Telegram Bot Tokeninizi Buraya Yazın
TOKEN = "8887451053:AAHszl4Q53MGxdv5cXETLEuE4IHDxq3jgEo"
bot = telebot.TeleBot(TOKEN)

PORTFOY_DOSYASI = "portfoy_data.json"
ALARMLAR_DOSYASI = "alarmlar_data.json"

# Pik Fiyat Takip ve Spam Önleme Hafızası
PEAK_PRICES = {}
NOTIFIED_STOCKS = {}

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
    try:
        with open(dosya_adi, "w", encoding="utf-8") as f:
            json.dump(veri, f, ensure_ascii=False, indent=4)
    except:
        pass


# 🛡️ GÜVENLİ FİYAT ÇEKME METODU
def guncel_fiyat_bul_bot(ticker_name):
    try:
        hisse = yf.Ticker(f"{ticker_name}.IS")
        df = hisse.history(period="5d")
        if not df.empty and "Close" in df.columns:
            gecerli_fiyatlar = df["Close"].dropna()
            if not gecerli_fiyatlar.empty:
                return round(float(gecerli_fiyatlar.iloc[-1]), 2)
        return None
    except:
        return None


def yapay_zeka_haber_analizi(ticker_obj):
    try:
        haberler = ticker_obj.news
        if not haberler:
            return 0
        pozitif_kelimeler = [
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
            "satin alma",
        ]
        negatif_kelimeler = [
            "zarar",
            "dusus",
            "kayip",
            "dava",
            "ceza",
            "iptal",
            "borc",
            "temerrut",
            "negatif",
            "azalma",
            "satis",
            "risk",
            "uyari",
        ]
        ai_skor = 0
        for h in haberler[:3]:
            baslik = h.get("title", "").lower()
            for p in pozitif_kelimeler:
                if p in baslik:
                    ai_skor += 1
            for n in negatif_kelimeler:
                if n in baslik:
                    ai_skor -= 1
        if ai_skor >= 2:
            return 2
        elif ai_skor <= -2:
            return -2
        return 0
    except:
        return 0


def canavar_teknik_analiz(df):
    if len(df) < 50:
        return 0
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    onay_sayisi = 0

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    if 35 < rsi.iloc[-1] < 65:
        onay_sayisi += 1

    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    if macd.iloc[-1] > signal.iloc[-1]:
        onay_sayisi += 1

    # Bollinger
    sma20 = close.rolling(window=20).mean()
    if close.iloc[-1] > sma20.iloc[-1]:
        onay_sayisi += 1

    # Stokastik
    low14 = low.rolling(window=14).min()
    high14 = high.rolling(window=14).max()
    k_percent = 100 * ((close - low14) / (high14 - low14 + 1e-10))
    d_percent = k_percent.rolling(window=3).mean()
    if k_percent.iloc[-1] > d_percent.iloc[-1] and k_percent.iloc[-1] < 80:
        onay_sayisi += 1

    # Ek Göstergeler
    if close.iloc[-1] > close.rolling(window=50).mean().iloc[-1]:
        onay_sayisi += 1
    if close.iloc[-1] > close.ewm(span=9, adjust=False).mean().iloc[-1]:
        onay_sayisi += 1
    if (
        volume.iloc[-1] > volume.rolling(window=10).mean().iloc[-1]
        and close.iloc[-1] > close.iloc[-2]
    ):
        onay_sayisi += 1
    if close.iloc[-1] > close.rolling(window=10).mean().iloc[-1]:
        onay_sayisi += 1

    return onay_sayisi


# 🎯 PİK SEVİYE VE İZ SÜREN STOP KONTROL DÖNGÜSÜ
def pik_fiyat_takip_dongusu():
    global PEAK_PRICES, NOTIFIED_STOCKS
    while True:
        try:
            portfoy = veri_yukle(PORTFOY_DOSYASI, {})
            if portfoy:
                kullanicilar = veri_yukle("aktif_kullanicilar.json", [])
                for ticker in portfoy.keys():
                    guncel_fiyat = guncel_fiyat_bul_bot(ticker)
                    if guncel_fiyat is None:
                        continue

                    # Zirve fiyatı güncelle
                    if (
                        ticker not in PEAK_PRICES
                        or guncel_fiyat > PEAK_PRICES[ticker]
                    ):
                        PEAK_PRICES[ticker] = guncel_fiyat
                        NOTIFIED_STOCKS[ticker] = (
                            False  # Yeni zirve geldiyse bildirimi sıfırla
                        )

                    zirve_fiyat = PEAK_PRICES[ticker]
                    dusus_yuzdesi = (
                        ((zirve_fiyat - guncel_fiyat) / zirve_fiyat) * 100
                        if zirve_fiyat > 0
                        else 0
                    )

                    # Pikten %1.5 düşüş ve henüz bildirim atılmadıysa
                    if (
                        dusus_yuzdesi >= 1.5
                        and not NOTIFIED_STOCKS.get(ticker, False)
                    ):
                        mesaj = (
                            "🚨 *CANAVAR AI PİK SAT UYARISI!*\n\n"
                            f"📌 *Hisse:* {ticker}\n"
                            f"🔥 *Anlık Fiyat:* {guncel_fiyat:.2f} TL\n"
                            f"🔝 *Gördüğü Zirve:* {zirve_fiyat:.2f} TL\n"
                            f"📉 *Pikten Düşüş:* %{dusus_yuzdesi:.2f}\n\n"
                            "⚠️ Kârı korumak için satışı/stop-loss'u"
                            " değerlendirin!\n"
                            "───────────────────"
                        )

                        for cid in kullanicilar:
                            try:
                                bot.send_message(
                                    cid, mesaj, parse_mode="Markdown"
                                )
                            except:
                                pass

                        NOTIFIED_STOCKS[ticker] = (
                            True  # Tekrar tekrar mesaj atıp darlamasın
                        )
        except Exception as e:
            pass

        time.sleep(60)  # Her 60 saniyede bir kontrol et


# 🚨 CANLI ALARM KONTROL DÖNGÜSÜ (60 Saniyede Bir Sorgular)
def alarm_kontrol_dongusu():
    while True:
        try:
            alarmlar = veri_yukle(ALARMLAR_DOSYASI, [])
            aktif_alarmlar = [a for a in alarmlar if a.get("durum") == "AKTİF"]

            if aktif_alarmlar:
                for alarm in aktif_alarmlar:
                    ticker = alarm.get("hisse")
                    hedef_fiyat = float(alarm.get("fiyat", 0))
                    yon = alarm.get("yon", "GEÇİNCE")

                    guncel_fiyat = guncel_fiyat_bul_bot(ticker)
                    if guncel_fiyat is not None:
                        tetiklendi = False
                        if yon == "GEÇİNCE" and guncel_fiyat >= hedef_fiyat:
                            tetiklendi = True
                        elif yon == "DÜŞÜNCE" and guncel_fiyat <= hedef_fiyat:
                            tetiklendi = True

                        if tetiklendi:
                            mesaj = (
                                "🚨 *CANAVAR ALARM UYARISI!*\n\n"
                                f"📈 *{ticker}* hissesi hedefe ulaştı!\n"
                                f"🎯 *Hedef Fiyat:* {hedef_fiyat:.2f} TL\n"
                                f"🔥 *Anlık Fiyat:* {guncel_fiyat:.2f} TL\n"
                                "───────────────────"
                            )

                            alarm["durum"] = "TETİKLENDİ"
                            veri_kaydet(ALARMLAR_DOSYASI, alarmlar)

                            kullanicilar = veri_yukle(
                                "aktif_kullanicilar.json", []
                            )
                            for cid in kullanicilar:
                                try:
                                    bot.send_message(
                                        cid, mesaj, parse_mode="Markdown"
                                    )
                                except:
                                    pass
        except Exception as e:
            pass

        time.sleep(60)


@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    kullanicilar = veri_yukle("aktif_kullanicilar.json", [])
    if message.chat.id not in kullanicilar:
        kullanicilar.append(message.chat.id)
        veri_kaydet("aktif_kullanicilar.json", kullanicilar)

    bot.reply_to(
        message,
        "🛡️ *Canavar AI Bot v3.3*\n\n"
        "/tara - Piyasayı tara\n"
        "/portfoy veya /portföy - Portföyüne bak",
    )


@bot.message_handler(commands=["tara"])
def command_tara(message):
    bot.reply_to(
        message,
        "⏳ 75 Lokomotif hisse ve güncel KAP/Haber akışları Yapay Zeka"
        " tarafından analiz ediliyor, lütfen bekleyin...",
    )

    canavar_al = []
    notr_bekle = []
    riskli_sat = []

    portfoy = veri_yukle(PORTFOY_DOSYASI, {})
    simdi = datetime.now().strftime("%H:%M:%S")

    for h in BIST_OTOMATIK_HAVUZ:
        if h in portfoy:
            continue
        try:
            hisse_obj = yf.Ticker(f"{h}.IS")
            df = hisse_obj.history(period="100d", interval="1d")
            tech_skor = canavar_teknik_analiz(df)

            if tech_skor >= 4:
                ai_puan = yapay_zeka_haber_analizi(hisse_obj)
                toplam_skor = tech_skor + ai_puan
                fiyat = guncel_fiyat_bul_bot(h)

                if fiyat:
                    if toplam_skor >= 6:
                        canavar_al.append(
                            f"🔥 *{h}* - {fiyat} TL (Skor: {toplam_skor}/8)"
                        )
                    elif toplam_skor >= 4:
                        notr_bekle.append(
                            f"🟡 *{h}* - {fiyat} TL (Skor: {toplam_skor}/8)"
                        )
                    else:
                        riskli_sat.append(
                            f"🔴 *{h}* - {fiyat} TL (Skor: {toplam_skor}/8)"
                        )
        except:
            pass

    cevap = f"📊 *CANAVAR AI HİBRİT TARAMA RAPORU*\n({simdi})\n"
    cevap += "───────────────────\n\n"

    cevap += "🟢 *🔥 CANAVAR AL SİNYALLERİ*\n"
    if canavar_al:
        cevap += "\n".join(canavar_al[:8]) + "\n"
    else:
        cevap += "_Uyan hisse bulunamadı._\n"
    cevap += "───────────────────\n\n"

    cevap += "🛑 *VETO EDİLEN / RİSKLİ / SAT SİNYALLERİ*\n"
    if riskli_sat:
        cevap += "\n".join(riskli_sat[:5]) + "\n"
    else:
        cevap += "_Filtreye takılan riskli durum yok._\n"
    cevap += "───────────────────\n\n"

    cevap += "🟡 *NÖTR / İZLEMEDE KALANLAR*\n"
    if notr_bekle:
        cevap += "\n".join(notr_bekle[:5]) + "\n"
    else:
        cevap += "_İzlenecek nötr hisse yok._\n"

    bot.send_message(message.chat.id, cevap, parse_mode="Markdown")


@bot.message_handler(commands=["portfoy", "portföy"])
def command_portfoy(message):
    kullanicilar = veri_yukle("aktif_kullanicilar.json", [])
    if message.chat.id not in kullanicilar:
        kullanicilar.append(message.chat.id)
        veri_kaydet("aktif_kullanicilar.json", kullanicilar)

    portfoy = veri_yukle(PORTFOY_DOSYASI, {})
    if not portfoy:
        bot.reply_to(
            message,
            "📭 Portföyünüz boş veya veri bulunamadı. Lütfen web sitesinden"
            " hisse ekleyin.",
        )
        return

    bot.reply_to(
        message,
        "⏳ Portföyünüzdeki hisselerin anlık canlı verileri çekiliyor...",
    )

    cevap = "💼 *ANLIK PORTFÖY DURUM RAPORU*\n"
    cevap += "───────────────────\n\n"
    toplam_deger = 0
    toplam_maliyet_genel = 0
    toplam_kar_zarar = 0

    for ticker, bilgi in portfoy.items():
        try:
            guncel_fiyat = guncel_fiyat_bul_bot(ticker)
            adet = float(bilgi.get("adet", 0))
            maliyet = float(bilgi.get("maliyet", 0))

            if guncel_fiyat is None:
                guncel_fiyat = maliyet

            if adet <= 0:
                continue

            toplam_maliyet = maliyet * adet
            guncel_deger = guncel_fiyat * adet
            kz = guncel_deger - toplam_maliyet
            kz_yuzde = (
                (kz / toplam_maliyet) * 100 if toplam_maliyet > 0 else 0
            )

            toplam_deger += guncel_deger
            toplam_maliyet_genel += toplam_maliyet
            toplam_kar_zarar += kz

            durum_oku = "🔺" if kz >= 0 else "🔻"
            cevap += f"📈 *{ticker}* | Adet: {adet:.0f}\n"
            cevap += (
                f"   Maliyet: {maliyet:.2f} TL ➔ Güncel: {guncel_fiyat:.2f} TL\n"
            )
            cevap += (
                f"   Net K/Z: {durum_oku} {kz:+.2f} TL ({kz_yuzde:+.2f}%)\n\n"
            )
        except Exception as e:
            pass

    if toplam_maliyet_genel > 0:
        toplam_yuzde_kz = (toplam_kar_zarar / toplam_maliyet_genel) * 100
        durum_genel = "🟢" if toplam_kar_zarar >= 0 else "🔴"

        cevap += "───────────────────\n"
        cevap += f"🏆 *Toplam Değer:* {toplam_deger:.2f} TL\n"
        cevap += (
            f"📊 *Net Kâr/Zarar:* {durum_genel} {toplam_kar_zarar:+.2f} TL"
            f" ({toplam_yuzde_kz:+.2f}%)\n"
        )
    else:
        cevap += "Portföy değerleri hesaplanamadı."

    bot.send_message(message.chat.id, cevap, parse_mode="Markdown")


if __name__ == "__main__":
    # 1. Thread: Fiyat Alarmlarını İzler
    t1 = threading.Thread(target=alarm_kontrol_dongusu)
    t1.daemon = True
    t1.start()

    # 2. Thread: Portföydeki Hisselerin Zirve Düşüşünü (Pik SAT) İzler
    t2 = threading.Thread(target=pik_fiyat_takip_dongusu)
    t2.daemon = True
    t2.start()

    bot.infinity_polling()
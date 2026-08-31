# EXP3 — Skenario 3: Passive + Active Attack (Pilot Contamination)

## Deskripsi

Eksperimen ini merupakan skenario serangan aktif (pilot contamination) oleh Eve
terhadap sistem PLKG (Physical Layer Key Generation) berbasis mesh batman-adv.

Eve menjalankan dua peran secara bersamaan:
1. **Passive** — tshark di `wlan0mon` merekam seluruh traffic probe mesh
2. **Active** — menyuntikkan frame palsu yang menyamar sebagai Alice

---

## Topologi

```
     [Alice] ec:08:6b:13:df:df
      /    \
  AB/9001  AC/9002
    /        \
[Bob]──BC/9003──[Charlie]
c4:e9:84:0f:1d:16   60:e3:27:0d:b1:e6

[Eve] wlan0mon ch.6 (monitor mode)
  → passive: rekam semua port 9001/9002/9003/9999
  → active:  inject frame spoofing MAC Alice
```

---

## Skema Active Attack

### Metode: Continuous Pilot Contamination

Eve menyuntikkan frame 802.11 data yang memalsukan MAC address Alice
(`ec:08:6b:13:df:df`) secara kontinu selama sesi probing berlangsung.

```python
inject_frame = (
    RadioTap() /
    Dot11(type=2, subtype=0,
          addr1='ff:ff:ff:ff:ff:ff',   # broadcast
          addr2='ec:08:6b:13:df:df',   # spoof Alice MAC
          addr3='ff:ff:ff:ff:ff:ff') /
    Raw(load=b'\x00' * 100)
)
# Inject terus menerus @ ~2/detik selama 310 detik
```

**Efek:** Bob dan Charlie menerima frame palsu dari "Alice". RSSI yang tercatat
di `iw dev wlan1 station dump` untuk Alice terkontaminasi oleh sinyal Eve,
bukan sinyal Alice yang sesungguhnya. Hal ini menyebabkan Alice dan Bob/Charlie
mengukur RSSI yang berbeda → key tidak cocok (KDR naik).

---

## Kenapa Tidak Pakai Trigger UDP Port

Script asli (`eve_collect.py`) menggunakan `PROBE_PORT = 9999` untuk mendeteksi
kapan Alice mengirim probe, lalu inject pada momen tersebut.

**Masalah:** batman-adv mengenkapsulasi UDP di dalam header proprietary:

```
802.11 frame → batman-adv header → IP → UDP(9999)
```

Scapy tidak memiliki dissector batman-adv, sehingga `UDP not in pkt` selalu
`True` → tidak ada injeksi yang terjadi (0 injections).

tshark berhasil karena memiliki dissector batman-adv built-in.

### Solusi: Continuous Injection (tanpa trigger)

Karena scapy tidak bisa melihat UDP layer, trigger diganti dengan loop kontinu:
- Tidak perlu deteksi paket probe
- Inject setiap ~0.5 detik sepanjang durasi eksperimen
- Bekerja di semua jenis enkapsulasi

**Justifikasi untuk paper:** Serangan kontinu lebih realistis karena attacker
nyata tidak perlu sinkronisasi sempurna dengan probe timing.

---

## Parameter Eksperimen

| Parameter | Nilai |
|-----------|-------|
| Samples | 2000 per link |
| Interval | 0.11 s |
| Link AB port | 9001 |
| Link AC port | 9002 |
| Link BC port | 9003 |
| Probe umpan port | 9999 |
| Eve inject rate | ~2 frame/detik |
| Eve duration | 310 detik |
| Total injections | **598** |

---

## Output Files

| File | Keterangan |
|------|------------|
| `tshark_alice_skenario3.csv` | Radiotap RSSI di Alice (mon0) selama attack |
| `tshark_bob_skenario3.csv` | Radiotap RSSI di Bob (mon0) selama attack |
| `tshark_charlie_skenario3.csv` | Radiotap RSSI di Charlie (mon0) selama attack |
| `tshark_eve_skenario3.csv` | Passive RSSI Eve (semua port 9001/9002/9003/9999) |
| `eve_attack_mac_skenario3.csv` | Log injeksi: timestamp + total injections |
| `eve_attack_mac_skenario3.log` | stdout log proses attack |

---

## Hasil

```
[attack] Done: 598 injections in 310s
AB: 1514/2000 valid probes
BC: 1322/2000 valid probes
Eve passive: 2.36 MB captured
```

---

## Relevansi untuk Paper IEEE

Data exp3 digunakan untuk membuktikan:

1. **Serangan berhasil** — 598 frame palsu disuntikkan, RSSI Alice di Bob/Charlie
   terkontaminasi
2. **Cross-link detection** — link AB/BC terkontaminasi, AC tidak → inkonsistensi
   terdeteksi dari perbandingan antar link
3. **Defense** — sistem dapat menolak pengukuran yang terkontaminasi menggunakan
   cross-link consistency check, sehingga KDR tetap rendah

Bandingkan dengan **exp1** dan **exp2** (kondisi normal, tanpa serangan) untuk
mengukur kenaikan KDR akibat pilot contamination.

---

## Script yang Digunakan

| Script | Lokasi | Fungsi |
|--------|--------|--------|
| `run_skenario3.py` | laptop | Koordinator utama |
| `orchestrator.py` | laptop | Jalankan 3 link serentak + tshark |
| `eve_attack_mac.py` | Eve `/home/alice/plkg/` | Continuous injection |
| `channel_probe.py` | Alice/Bob/Charlie `/home/riyadhX/plkg/` | UDP probe |

# EXP1 — Skema Channel Probing Antar Node (Baseline)

## Deskripsi

Eksperimen baseline PLKG (Physical Layer Key Generation) pada topologi mesh
batman-adv 3 node. Semua link diprobing secara **serentak** (simultaneous),
sesuai prinsip mesh di mana semua jalur aktif pada saat yang sama.

---

## Topologi Mesh

```
         [Alice]  10.0.0.1
          /    \
      AB/9001  AC/9002
        /        \
    [Bob]──BC/9003──[Charlie]
   10.0.0.2        10.0.0.3

Eve: wlan0mon ch.6 (monitor mode, passive)
```

---

## Skema Channel Probing

### Prinsip Dasar

Setiap pasangan node (Alice-Bob, Alice-Charlie, Bob-Charlie) saling mengukur
kekuatan sinyal (RSSI) secara **dua arah** menggunakan UDP probe packets.

```
Alice ──[UDP probe]──► Bob
Bob   ──[UDP reply]──► Alice
(RSSI Alice dari Bob = RSSI Bob dari Alice → reciprocity)
```

Reciprocity ini yang menjadi basis key generation: kedua pihak mengukur
channel yang sama, sehingga menghasilkan bit yang identik.

### Peran Node

| Peran | Fungsi |
|-------|--------|
| **Coordinator** | Mengirim probe, menerima reply, mencatat RSSI lokal + remote |
| **Reflector** | Menerima probe, echo balik dengan RSSI lokal |

### Alokasi Port per Link (simultaneous, tanpa konflik)

| Link | Coordinator | Reflector | Port UDP |
|------|-------------|-----------|----------|
| AB | Alice | Bob | 9001 |
| AC | Alice | Charlie | 9002 |
| BC | Bob | Charlie | 9003 |

### Urutan Eksekusi (serentak)

```
t=0s  │ Alice  : start tshark mon0, start reflector AB+AC
      │ Bob    : start tshark mon0, start reflector BC
      │ Charlie: start tshark mon0
      │
t=2s  │ [semua 3 link mulai bersamaan dalam thread terpisah]
      │ Thread-1: Alice coordinator → Bob port 9001
      │ Thread-2: Alice coordinator → Charlie port 9002
      │ Thread-3: Bob coordinator → Charlie port 9003
      │
      │ Eve: tshark wlan0mon, passive capture semua port
      │
t+220s│ Semua coordinator selesai (2000 × 0.11s)
      │
t+225s│ Stop tshark semua node → download CSV
```

### Format Data Probe (channel_probe.py)

**Coordinator kirim:**
```
struct: !Id? → (seq: uint32, timestamp: double, is_reply: bool)
```

**Reflector reply:**
```
struct: !IddB → (seq: uint32, ts_orig: double, rssi_local: double, flag: uint8)
```

**Output CSV koordinator:**
```
seq, timestamp, rssi_local, rssi_remote, rtt_ms
```

---

## RSSI Collection: Dua Metode

### Metode 1 — iw station dump (per-session average)
```bash
iw dev wlan1 station dump | grep "last signal"
```
- Dibaca setiap kali coordinator menerima reply
- RSSI = sinyal terakhir dari peer di interface wlan1
- Resolusi: ~1 dBm

### Metode 2 — tshark radiotap (per-packet, lebih akurat)
```bash
tshark -i mon0 -Y "udp.port==9001 or udp.port==9002 or udp.port==9003" \
  -T fields -e frame.time_epoch -e radiotap.dbm_antsignal \
  -e wlan.sa -e wlan.da -e ip.src -e ip.dst \
  -e udp.srcport -e udp.dstport
```
- Ditangkap dari virtual monitor interface `mon0`
- mon0 dibuat dari wlan1 tanpa melepaskan mesh (`iw dev wlan1 interface add mon0 type monitor`)
- RSSI per-packet dari radiotap header (lebih akurat dari station dump)
- **Ini sumber data utama untuk keygen dan analisis paper**

---

## Output Files

| File | Node | Keterangan |
|------|------|------------|
| `tshark_alice_exp04.csv` | Alice | Radiotap RSSI semua frame probe di Alice |
| `tshark_bob_exp04.csv` | Bob | Radiotap RSSI semua frame probe di Bob |
| `tshark_charlie_exp04.csv` | Charlie | Radiotap RSSI semua frame probe di Charlie |
| `tshark_eve_exp04.csv` | Eve | Passive RSSI (monitor mode, sudut pandang attacker) |

### Format CSV

```
frame.time_epoch, radiotap.dbm_antsignal, wlan.sa, wlan.da,
ip.src, ip.dst, udp.srcport, udp.dstport
```

---

## Parameter Eksperimen

| Parameter | Nilai |
|-----------|-------|
| Samples per link | 1000 (exp04) |
| Interval probe | 0.11 s |
| Total durasi | ~110 s per link (~110 s total, serentak) |
| Channel WiFi | 6 (2437 MHz) |
| Mesh protocol | batman-adv over 802.11s |
| Interface probe | wlan1 (mesh point) |
| Interface capture | mon0 (virtual monitor dari wlan1) |
| Eve interface | wlan0mon (monitor mode, AR9271 dongle) |

---

## Kondisi Eksperimen

- **Lingkungan:** Indoor
- **Eve mode:** Passive only (tidak ada injeksi)
- **Posisi Eve:** P3 (tengah segitiga)
- **Tujuan:** Baseline — ukur KGR, KDR, dan Eve BER dalam kondisi normal

"""
Pemetaan nama lokasi (kota/kabupaten/provinsi Indonesia) -> koordinat,
untuk peta sebaran lowongan. Nilai: (Nama tampil, lat, lng).
"""
import re

COORDS = {
    # DKI & sekitarnya
    "jakarta": ("Jakarta", -6.2088, 106.8456),
    "tangerang selatan": ("Tangerang Selatan", -6.2889, 106.7180),
    "tangerang": ("Tangerang", -6.1783, 106.6319),
    "bekasi": ("Bekasi", -6.2383, 106.9756),
    "depok": ("Depok", -6.4025, 106.7942),
    "bogor": ("Bogor", -6.5950, 106.8166),
    "cikarang": ("Cikarang", -6.2614, 107.1525),
    "karawang": ("Karawang", -6.3227, 107.3376),
    # Jawa Barat / Banten
    "bandung": ("Bandung", -6.9175, 107.6191),
    "cimahi": ("Cimahi", -6.8722, 107.5424),
    "cirebon": ("Cirebon", -6.7320, 108.5523),
    "sukabumi": ("Sukabumi", -6.9277, 106.9300),
    "tasikmalaya": ("Tasikmalaya", -7.3274, 108.2207),
    "purwakarta": ("Purwakarta", -6.5569, 107.4439),
    "serang": ("Serang", -6.1100, 106.1503),
    "cilegon": ("Cilegon", -6.0174, 106.0541),
    "banten": ("Banten", -6.4058, 106.0640),
    "jawa barat": ("Jawa Barat", -6.9147, 107.6098),
    # Jawa Tengah & DIY
    "semarang": ("Semarang", -6.9667, 110.4167),
    "yogyakarta": ("Yogyakarta", -7.7956, 110.3695),
    "sleman": ("Sleman", -7.7167, 110.3556),
    "bantul": ("Bantul", -7.8881, 110.3298),
    "surakarta": ("Surakarta (Solo)", -7.5755, 110.8243),
    "solo": ("Surakarta (Solo)", -7.5755, 110.8243),
    "magelang": ("Magelang", -7.4797, 110.2177),
    "salatiga": ("Salatiga", -7.3305, 110.5084),
    "tegal": ("Tegal", -6.8694, 109.1402),
    "pekalongan": ("Pekalongan", -6.8898, 109.6753),
    "kudus": ("Kudus", -6.8048, 110.8405),
    "cilacap": ("Cilacap", -7.7259, 109.0150),
    "purwokerto": ("Purwokerto", -7.4216, 109.2347),
    "banyumas": ("Banyumas", -7.4216, 109.2347),
    "jawa tengah": ("Jawa Tengah", -7.1500, 110.1403),
    # Jawa Timur
    "surabaya": ("Surabaya", -7.2575, 112.7521),
    "sidoarjo": ("Sidoarjo", -7.4478, 112.7183),
    "gresik": ("Gresik", -7.1556, 112.6516),
    "malang": ("Malang", -7.9666, 112.6326),
    "kediri": ("Kediri", -7.8480, 112.0178),
    "jember": ("Jember", -8.1724, 113.7002),
    "banyuwangi": ("Banyuwangi", -8.2192, 114.3691),
    "probolinggo": ("Probolinggo", -7.7543, 113.2159),
    "mojokerto": ("Mojokerto", -7.4722, 112.4338),
    "pasuruan": ("Pasuruan", -7.6453, 112.9075),
    "madiun": ("Madiun", -7.6298, 111.5239),
    "jawa timur": ("Jawa Timur", -7.5361, 112.2384),
    # Bali & Nusa Tenggara
    "denpasar": ("Denpasar (Bali)", -8.6705, 115.2126),
    "bali": ("Bali", -8.4095, 115.1889),
    "badung": ("Badung (Bali)", -8.5800, 115.1780),
    "mataram": ("Mataram (Lombok)", -8.5833, 116.1167),
    "lombok": ("Lombok", -8.6500, 116.3249),
    "kupang": ("Kupang", -10.1772, 123.6070),
    # Sumatera
    "medan": ("Medan", 3.5952, 98.6722),
    "sumatera utara": ("Sumatera Utara", 2.1154, 99.5451),
    "palembang": ("Palembang", -2.9761, 104.7754),
    "pekanbaru": ("Pekanbaru", 0.5071, 101.4478),
    "riau": ("Riau", 0.5071, 101.4478),
    "padang": ("Padang", -0.9471, 100.4172),
    "batam": ("Batam", 1.0456, 104.0305),
    "jambi": ("Jambi", -1.6101, 103.6131),
    "lampung": ("Lampung", -5.4294, 105.2610),
    "bandar lampung": ("Bandar Lampung", -5.3971, 105.2668),
    "bengkulu": ("Bengkulu", -3.7928, 102.2608),
    "aceh": ("Aceh", 5.5483, 95.3238),
    "banda aceh": ("Banda Aceh", 5.5483, 95.3238),
    # Kalimantan
    "balikpapan": ("Balikpapan", -1.2379, 116.8529),
    "samarinda": ("Samarinda", -0.5022, 117.1536),
    "pontianak": ("Pontianak", -0.0263, 109.3425),
    "banjarmasin": ("Banjarmasin", -3.3186, 114.5944),
    "palangkaraya": ("Palangka Raya", -2.2096, 113.9136),
    "kalimantan timur": ("Kalimantan Timur", -1.2379, 116.8529),
    "kalimantan barat": ("Kalimantan Barat", -0.0263, 109.3425),
    "kalimantan selatan": ("Kalimantan Selatan", -3.3186, 114.5944),
    # Sulawesi
    "makassar": ("Makassar", -5.1477, 119.4327),
    "manado": ("Manado", 1.4748, 124.8421),
    "palu": ("Palu", -0.8917, 119.8707),
    "kendari": ("Kendari", -3.9778, 122.5150),
    "gorontalo": ("Gorontalo", 0.5435, 123.0568),
    "sulawesi selatan": ("Sulawesi Selatan", -5.1477, 119.4327),
    # Maluku & Papua
    "ambon": ("Ambon", -3.6954, 128.1814),
    "jayapura": ("Jayapura", -2.5916, 140.6690),
    "papua": ("Papua", -4.2699, 138.0804),
    "sorong": ("Sorong", -0.8762, 131.2558),
}

_KEYS_BY_LEN = sorted(COORDS, key=len, reverse=True)


def resolve(loc):
    """Kembalikan (nama, lat, lng) untuk string lokasi, atau None bila tak dikenali."""
    if not loc:
        return None
    loc = str(loc).lower().strip()
    if loc in COORDS:
        return COORDS[loc]
    tokens = set(t for t in re.split(r"[^a-z]+", loc) if t)
    for key in _KEYS_BY_LEN:
        if " " in key:
            if key in loc:
                return COORDS[key]
        elif key in tokens:
            return COORDS[key]
    return None

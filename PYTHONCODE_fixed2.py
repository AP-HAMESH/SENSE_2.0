"""
EEG Live Monitor — V5 (Fixed)
- Single waveform (filtered only)
- Y-axis scale slider
- ML N/A shown clearly when no model loaded
"""

import sys
import time
import os
import socket
import struct
import json
import csv
import argparse
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi, welch, iirnotch, filtfilt
from scipy.stats import kurtosis as sp_kurtosis
import serial
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

# =============================================================================
# CLI ARGUMENTS
# =============================================================================
parser = argparse.ArgumentParser(description="EEG Live Monitor V5 Fixed")
parser.add_argument('--port',        default='COM11')
parser.add_argument('--baud',        default=115200,   type=int)
parser.add_argument('--bits',        default=8,        type=int, choices=[8, 10, 12, 16, 24])
parser.add_argument('--baseline-sec', default=60,       type=int,
                    help="Seconds of good data needed before state classification starts. "
                         "Lower this (e.g. 15-20) while debugging to iterate faster.")
parser.add_argument('--no-matlab',   action='store_true')
parser.add_argument('--focus-model', default=None)
parser.add_argument('--nband-model', default=None)
parser.add_argument('--nband-labels',
                    default='VALENCE_LOW,VALENCE_HIGH,AROUSAL_LOW,AROUSAL_HIGH')
parser.add_argument('--nband-csv',   default=None)
parser.add_argument('--scaler',      default=None)
args = parser.parse_args()

PORT           = args.port
BAUD           = args.baud
BITS           = args.bits
SEND_TO_MATLAB = not args.no_matlab

# =============================================================================
# CONFIG
# =============================================================================
FS                  = 250
DISPLAY_SEC         = 10
BUFFER_SIZE         = FS * DISPLAY_SEC

BASELINE_SEC        = args.baseline_sec
BASELINE_SAMPLES    = FS * BASELINE_SEC

PSD_UPDATE_INTERVAL = 25
WELCH_NPERSEG       = int(FS * 2)
ML_EPOCH_SAMPLES    = int(FS * 1)

MATLAB_UDP_IP       = '172.16.219.144'
MATLAB_UDP_PORT     = 5005
DRIFT_WARN_THRESH   = 0.05

ADC_FULL_SCALE      = 2 ** BITS
ARTIFACT_THRESHOLD  = int(ADC_FULL_SCALE * 0.15)

# Y-scale default (±counts) — user can change via slider
y_scale = 300

# =============================================================================
# BAND DEFINITIONS
# =============================================================================
BANDS = {
    'delta': (3.5,  4.0),
    'theta': (4.0,  6.0),
    'alpha': (6.0, 11.0),
    'beta':  (11.0, 30.0),
}
POWER_BAND_TOTAL = (3.5, 28.0)

BAND_COLORS = {
    'delta': '#4A90D9',
    'theta': '#7B68EE',
    'alpha': '#2ECC71',
    'beta':  '#E74C3C',
}

# =============================================================================
# BLINK DETECTOR CONFIG
# =============================================================================
BLINK_WINDOW_SEC    = 0.4
BLINK_WINDOW_SAMP   = int(FS * BLINK_WINDOW_SEC)
BLINK_STEP_SAMP     = int(FS * 0.04)
BLINK_PP_THRESHOLD  = ADC_FULL_SCALE * 0.08
BLINK_KURTOSIS_MIN  = 2.5
BLINK_INTERP_PAD    = int(FS * 0.05)

# =============================================================================
# ML MODEL LOADING
# =============================================================================
_focus_interpreter = None
_focus_input_idx   = None
_focus_output_idx  = None

def _load_focus_model(path):
    global _focus_interpreter, _focus_input_idx, _focus_output_idx
    if not path:
        return
    try:
        import tensorflow as tf
        if path.endswith('.tflite'):
            interp = tf.lite.Interpreter(model_path=path)
            interp.allocate_tensors()
            _focus_interpreter = interp
            _focus_input_idx   = interp.get_input_details()[0]['index']
            _focus_output_idx  = interp.get_output_details()[0]['index']
            print(f"[ML-1] Loaded TFLite: {path}")
        else:
            _focus_interpreter = tf.keras.models.load_model(path) \
                if not os.path.isdir(path) else tf.saved_model.load(path)
            print(f"[ML-1] Loaded Keras: {path}")
    except Exception as e:
        print(f"[ML-1] Load failed: {e}")

_load_focus_model(args.focus_model)

def predict_focus(ratio_features):
    if _focus_interpreter is None:
        return "No model loaded", 0.0
    try:
        import tensorflow as tf
        feat = ratio_features.astype(np.float32).reshape(1, 4)
        if isinstance(_focus_interpreter, tf.lite.Interpreter):
            _focus_interpreter.set_tensor(_focus_input_idx, feat)
            _focus_interpreter.invoke()
            out   = _focus_interpreter.get_tensor(_focus_output_idx)[0]
            score = float(out[0]) if out.shape[0] == 1 else float(out[1])
        else:
            out   = _focus_interpreter(feat)
            score = float(out[0][0]) if out.shape[-1] == 1 else float(out[0][1])
        return ("High Focus" if score >= 0.5 else "Low Focus"), score
    except Exception as e:
        print(f"[ML-1] Inference error: {e}")
        return "Error", 0.0

_nband_model  = None
_nband_labels = [s.strip() for s in args.nband_labels.split(',')]
_nband_scaler = None

def _load_nband_scaler():
    global _nband_scaler
    if args.scaler:
        try:
            import joblib
            _nband_scaler = joblib.load(args.scaler)
            print(f"[ML-2] Scaler loaded: {args.scaler}")
            return
        except Exception as e:
            print(f"[ML-2] Scaler load failed: {e}")
    if args.nband_csv:
        try:
            import pandas as pd
            from sklearn.preprocessing import StandardScaler
            df = pd.read_csv(args.nband_csv)
            sc = StandardScaler()
            sc.fit(df[['delta','theta','alpha','beta']].values)
            _nband_scaler = sc
            print(f"[ML-2] Scaler fitted from {args.nband_csv}")
        except Exception as e:
            print(f"[ML-2] CSV scaler failed: {e}")

def _load_nband_model(path):
    global _nband_model
    if not path:
        return
    try:
        import tensorflow as tf
        _nband_model = tf.keras.models.load_model(path) \
            if not os.path.isdir(path) else tf.saved_model.load(path)
        print(f"[ML-2] Loaded: {path}")
    except Exception as e:
        print(f"[ML-2] Load failed: {e}")

_load_nband_scaler()
_load_nband_model(args.nband_model)

def predict_nband(abs_powers):
    if _nband_model is None:
        return "No model loaded", 0.0
    try:
        feat = abs_powers.astype(np.float32).reshape(1, 4)
        if _nband_scaler is not None:
            feat = _nband_scaler.transform(feat).astype(np.float32)
        out   = _nband_model(feat) if callable(_nband_model) \
            else _nband_model.predict(feat, verbose=0)
        probs = np.array(out[0])
        idx   = int(np.argmax(probs))
        label = _nband_labels[idx] if idx < len(_nband_labels) else str(idx)
        return label, float(probs[idx])
    except Exception as e:
        print(f"[ML-2] Inference error: {e}")
        return "Error", 0.0

# =============================================================================
# ML-3: FEATURE EXTRACTION
# =============================================================================
def ml_bandpass(signal, fs):
    b, a = butter(4, [0.5/(fs/2), 40.0/(fs/2)], btype='band')
    return filtfilt(b, a, signal)

def ml_normalize(signal):
    std = np.std(signal)
    return (signal - np.mean(signal)) / (std + 1e-9)

def _bp_abs(signal, fs, band):
    freqs, psd = welch(signal, fs, nperseg=min(len(signal), WELCH_NPERSEG))
    idx = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.trapezoid(psd[idx], freqs[idx])) if np.any(idx) else 0.0

def extract_ml_features(epoch, fs):
    ep    = ml_bandpass(epoch, fs)
    ep    = ml_normalize(ep)
    delta = _bp_abs(ep, fs, (0.5, 4.0))
    theta = _bp_abs(ep, fs, (4.0, 8.0))
    alpha = _bp_abs(ep, fs, (8.0, 13.0))
    beta  = _bp_abs(ep, fs, (13.0, 30.0))
    abs_p = np.array([delta, theta, alpha, beta], dtype=np.float32)
    ratio = np.array([
        theta / (alpha + 1e-6),
        beta  / (alpha + 1e-6),
        delta / (theta + 1e-6),
        alpha / (beta  + 1e-6),
        ], dtype=np.float32)
    return ratio, abs_p

# =============================================================================
# FILTER CHAIN
# =============================================================================
def _make(b, a):
    return b, a, lfilter_zi(b, a) * 0.0

def make_hp(cutoff, fs, order=4):
    b, a = butter(order, cutoff/(fs/2), btype='high')
    return _make(b, a)

def make_notch(freq, fs, Q=30):
    b, a = iirnotch(freq/(fs/2), Q)
    return _make(b, a)

def make_lp(cutoff, fs, order=4):
    b, a = butter(order, cutoff/(fs/2), btype='low')
    return _make(b, a)

b_hp05, a_hp05, zi_hp05 = make_hp(0.3, FS)
b_notch, a_notch, zi_notch = make_notch(50, FS, Q=40)
b_notch2, a_notch2, zi_notch2 = make_notch(100, FS, Q=40)
b_lp,    a_lp,    zi_lp    = make_lp(30, FS)    
b_hp10,  a_hp10            = butter(4, 1.0/(FS/2), btype='high')

def filter_sample_display(raw_val):
    global zi_hp05, zi_notch, zi_notch2, zi_lp
    x = np.array([float(raw_val)])
    x, zi_hp05   = lfilter(b_hp05,  a_hp05,  x, zi=zi_hp05)
    x, zi_notch  = lfilter(b_notch, a_notch, x, zi=zi_notch)
    x, zi_notch2 = lfilter(b_notch2, a_notch2, x, zi=zi_notch2)
    x, zi_lp     = lfilter(b_lp,    a_lp,    x, zi=zi_lp)
    return float(x[0])

def filter_sample_display(raw_val):
    global zi_hp05, zi_notch, zi_lp
    x = np.array([float(raw_val)])
    x, zi_hp05   = lfilter(b_hp05,  a_hp05,  x, zi=zi_hp05)
    x, zi_notch  = lfilter(b_notch, a_notch, x, zi=zi_notch)
    x, zi_lp     = lfilter(b_lp,    a_lp,    x, zi=zi_lp)
    return float(x[0])

# =============================================================================
# BUFFERS
# =============================================================================
raw_buf         = deque([0.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
filt_buf        = deque([0.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
epoch_buf       = deque(maxlen=int(FS * 2))
blink_clean_buf = deque([0.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
ml_epoch_acc    = deque(maxlen=ML_EPOCH_SAMPLES)

# =============================================================================
# BLINK DETECTOR
# =============================================================================
class BlinkDetector:
    def __init__(self):
        self.blink_count   = 0
        self.last_blink_at = -999

    def clean(self, signal, sample_offset=0):
        cleaned    = signal.copy()
        blink_mask = np.zeros(len(signal), dtype=bool)
        n = len(signal)
        i = 0
        while i < n - BLINK_WINDOW_SAMP:
            window = signal[i:i + BLINK_WINDOW_SAMP]
            pp   = float(np.ptp(window))
            kurt = float(sp_kurtosis(window, fisher=True))
            if pp > BLINK_PP_THRESHOLD and kurt > BLINK_KURTOSIS_MIN:
                global_idx = sample_offset + i
                if global_idx - self.last_blink_at > int(FS * 0.6):
                    self.last_blink_at = global_idx
                    self.blink_count  += 1
                    start = max(0, i - BLINK_INTERP_PAD)
                    end   = min(n-1, i + BLINK_WINDOW_SAMP + BLINK_INTERP_PAD)
                    pre   = float(np.mean(signal[max(0,start-5):start])) if start > 5 else 0.0
                    post  = float(np.mean(signal[end:min(n,end+5)]))     if end < n-5 else 0.0
                    cleaned[start:end+1]    = np.linspace(pre, post, end-start+1)
                    blink_mask[start:end+1] = True
                    i = end + 1
                    continue
            i += BLINK_STEP_SAMP
        return cleaned, blink_mask

blink_detector = BlinkDetector()

# =============================================================================
# PSD + BAND POWER
# =============================================================================
def compute_psd_clean(buffer_deque):
    x = np.array(buffer_deque, dtype=float)
    if len(x) < WELCH_NPERSEG:
        return None, None
    x = filtfilt(b_hp10, a_hp10, x)
    f, p = welch(x, fs=FS, window='hann',
                 nperseg=WELCH_NPERSEG,
                 noverlap=int(WELCH_NPERSEG * 0.75),
                 scaling='density')
    return f, p

def bandpower_psd(freqs, psd, band):
    idx = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.trapezoid(psd[idx], freqs[idx])) if np.any(idx) else 0.0

def compute_powers(freqs, psd):
    total_idx = (freqs >= POWER_BAND_TOTAL[0]) & (freqs <= POWER_BAND_TOTAL[1])
    total_p   = float(np.trapezoid(psd[total_idx], freqs[total_idx]))
    if total_p <= 0:
        return None
    return {
        'delta_n': bandpower_psd(freqs, psd, BANDS['delta']) / total_p,
        'theta_n': bandpower_psd(freqs, psd, BANDS['theta']) / total_p,
        'alpha_n': bandpower_psd(freqs, psd, BANDS['alpha']) / total_p,
        'beta_n':  bandpower_psd(freqs, psd, BANDS['beta'])  / total_p,
    }

# =============================================================================
# ADAPTIVE BASELINE CLASSIFIER
# =============================================================================
class Baseline:
    def __init__(self, n_samples):
        self.target     = n_samples // PSD_UPDATE_INTERVAL
        self.alpha_buf  = []
        self.beta_buf   = []
        self.ready      = False
        self.alpha_mean = self.alpha_std = None
        self.beta_mean  = self.beta_std  = None

    def update(self, a, b):
        if self.ready:
            return
        self.alpha_buf.append(a)
        self.beta_buf.append(b)
        if len(self.alpha_buf) >= self.target:
            self.alpha_mean = float(np.mean(self.alpha_buf))
            self.alpha_std  = max(float(np.std(self.alpha_buf)), 0.01)
            self.beta_mean  = float(np.mean(self.beta_buf))
            self.beta_std   = max(float(np.std(self.beta_buf)), 0.01)
            self.ready = True
            print(f"\n[BASELINE] α={self.alpha_mean:.3f}±{self.alpha_std:.3f} | "
                  f"β={self.beta_mean:.3f}±{self.beta_std:.3f}\n")

    def classify(self, powers):
        if not self.ready:
            pct = len(self.alpha_buf) / self.target * 100
            return f"Calibrating… {pct:.0f}%"
        a_z = (powers['alpha_n'] - self.alpha_mean) / self.alpha_std
        b_z = (powers['beta_n']  - self.beta_mean)  / self.beta_std
        t, d, a = powers['theta_n'], powers['delta_n'], powers['alpha_n']
        if   a_z >  1.0 and b_z <  0.0:                      return "Relaxed"
        elif b_z >  1.0 and a_z <  0.0:                      return "Stressed / High Focus"
        elif b_z >  0.5 and -0.5 < a_z < 0.5:               return "Focused"
        elif t   > 0.35 and a < (self.alpha_mean * 0.7):     return "Drowsy"
        elif d   > 0.50:                                      return "Artifact / Movement"
        else:                                                  return "Neutral"

baseline = Baseline(BASELINE_SAMPLES)

# =============================================================================
# CLOCK DRIFT MONITOR
# =============================================================================
class DriftMonitor:
    def __init__(self, window=500):
        self.window = window
        self.ts     = deque(maxlen=window)

    def tick(self, hw_us=None):
        t = hw_us/1e6 if hw_us is not None else time.perf_counter()
        self.ts.append(t)
        if len(self.ts) == self.window:
            span = self.ts[-1] - self.ts[0]
            if span > 0:
                fs_m  = (self.window-1)/span
                drift = abs(fs_m - FS)/FS
                if drift > DRIFT_WARN_THRESH:
                    print(f"[DRIFT] {fs_m:.1f} Hz measured (expected {FS}, "
                          f"drift={drift*100:.1f}%)")
            self.ts.clear()

drift_mon = DriftMonitor()

# =============================================================================
# SERIAL
# =============================================================================
ser_lock = threading.Lock()
_ser = None

def get_serial():
    global _ser
    with ser_lock:
        if _ser is not None and _ser.is_open:
            return _ser
        try:
            # FIX: was timeout=2 -> blocked the GUI thread for up to 2s per
            # readline() call when no data was arriving, causing "Not Responding".
            _ser = serial.Serial(PORT, BAUD, timeout=0.05)
            print(f"[SERIAL] Connected: {PORT}")
            return _ser
        except serial.SerialException:
            return None

def parse_line(line):
    line = line.strip()
    if line.startswith("TS:"):
        try:
            parts = dict(p.split(':') for p in line.split(','))
            return int(parts['VAL']), int(parts['TS'])
        except Exception:
            return None, None
    try:
        return int(line), None
    except ValueError:
        return None, None

# =============================================================================
# UDP
# =============================================================================
udp_sock = None
if SEND_TO_MATLAB:
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Pass the filtered epoch instead of f50/p50
def send_to_matlab(state, powers, latest_eeg_chunk, blink_count, ml1_label, ml2_label):
    if not SEND_TO_MATLAB or udp_sock is None:
        return
    header = json.dumps({
        's': state, 'a': round(powers['alpha_n'],4),
        'b': round(powers['beta_n'],4), 't': round(powers['theta_n'],4),
        'd': round(powers['delta_n'],4), 'n': len(latest_eeg_chunk),
        'blinks': blink_count, 'ml1': ml1_label, 'ml2': ml2_label,
    }).encode()

    # Send the time-series floats
    payload = struct.pack(f'<{len(latest_eeg_chunk)}f', *latest_eeg_chunk)
    frame = struct.pack('<I', len(header)) + header + payload
    udp_sock.sendto(frame, (MATLAB_UDP_IP, MATLAB_UDP_PORT))

# =============================================================================
# CSV LOGGER
# =============================================================================
OUTPUT_DIR = Path(r"C:\Users\AP HAMESH\OneDrive\Desktop\AP\MIND\SENSE 2.0")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log_path  = OUTPUT_DIR / f"eeg_log_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
_csv_file = open(log_path, 'w', newline='')
_csv_w    = csv.writer(_csv_file)
_csv_w.writerow(['timestamp_s','state',
                 'delta_n','theta_n','alpha_n','beta_n',
                 'blinks_total',
                 'focus_label','focus_score',
                 'nband_label','nband_conf'])
_t0 = time.perf_counter()

def log_row(state, powers, blinks, ml1_label, ml1_score, ml2_label, ml2_conf):
    t = round(time.perf_counter() - _t0, 3)
    _csv_w.writerow([t, state,
                     round(powers['delta_n'],4), round(powers['theta_n'],4),
                     round(powers['alpha_n'],4), round(powers['beta_n'],4),
                     blinks,
                     ml1_label, round(ml1_score,4),
                     ml2_label, round(ml2_conf,4)])

# =============================================================================
# UI — PyQtGraph
# =============================================================================
app = QtWidgets.QApplication(sys.argv)
pg.setConfigOptions(antialias=True)

# ── Main window with central widget so we can add a slider ──────────────────
main_win = QtWidgets.QMainWindow()
main_win.setWindowTitle(f"EEG Monitor V5 | {PORT} | {BITS}-bit | ML: Focus + N-Band")
main_win.resize(1380, 900)

central = QtWidgets.QWidget()
main_win.setCentralWidget(central)
layout = QtWidgets.QVBoxLayout(central)
layout.setContentsMargins(4, 4, 4, 4)
layout.setSpacing(4)

# ── Scale slider row ─────────────────────────────────────────────────────────
slider_row = QtWidgets.QHBoxLayout()
lbl_scale  = QtWidgets.QLabel("Y-Scale (±):  300")
lbl_scale.setStyleSheet("color: white; font-size: 12px;")
lbl_scale.setFixedWidth(160)

# FIX (PyQt6): Qt.Horizontal -> Qt.Orientation.Horizontal
slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
slider.setMinimum(50)
slider.setMaximum(2000)
slider.setValue(300)
slider.setTickInterval(100)
slider.setToolTip("Adjust Y-axis amplitude range")
slider.setStyleSheet("""
    QSlider::groove:horizontal {
        background: #333; height: 6px; border-radius: 3px;
    }
    QSlider::handle:horizontal {
        background: #00BFFF; width: 16px; height: 16px;
        margin: -5px 0; border-radius: 8px;
    }
    QSlider::sub-page:horizontal { background: #00BFFF; border-radius: 3px; }
""")

def on_slider(val):
    global y_scale
    y_scale = val
    lbl_scale.setText(f"Y-Scale (±):  {val}")
    p_time.setYRange(-val, val)

slider.valueChanged.connect(on_slider)

btn_autoscale = QtWidgets.QPushButton("Auto-Scale")
btn_autoscale.setStyleSheet(
    "background:#2ECC71; color:black; font-weight:bold;"
    "padding:4px 12px; border-radius:4px;")
btn_autoscale.clicked.connect(lambda: p_time.enableAutoRange(axis='y'))

slider_row.addWidget(lbl_scale)
slider_row.addWidget(slider)
slider_row.addWidget(btn_autoscale)
layout.addLayout(slider_row)

# ── PyQtGraph layout ─────────────────────────────────────────────────────────
gw = pg.GraphicsLayoutWidget()
gw.setBackground('#0f0f12')
layout.addWidget(gw)

# ── Row 1: Waveform (filtered only — single curve) ───────────────────────────
p_time = gw.addPlot(title="EEG Signal  (filtered · 0.5 Hz HPF + Notch + 30 Hz LPF)",
                    colspan=2)
p_time.setLabel('left', 'Amplitude (ADC counts)')
p_time.setLabel('bottom', 'Samples →')
p_time.setYRange(-y_scale, y_scale)
p_time.showGrid(x=True, y=True, alpha=0.2)

curve_filt = p_time.plot(pen=pg.mkPen('#F4A261', width=1.4), name="Filtered EEG")

state_label = pg.TextItem(text="State: Calibrating…", color='w', anchor=(0,1))
# FIX (PyQt6): QFont.Bold -> QFont.Weight.Bold
state_label.setFont(QtGui.QFont('Arial', 13, QtGui.QFont.Weight.Bold))
p_time.addItem(state_label)
state_label.setPos(0, -y_scale * 0.85)

blink_label = pg.TextItem(text="Blinks: 0", color='#2ECC71', anchor=(1,1))
blink_label.setFont(QtGui.QFont('Arial', 11))
p_time.addItem(blink_label)
blink_label.setPos(BUFFER_SIZE, -y_scale * 0.85)

gw.nextRow()

# ── Row 2: PSD + Band bars ───────────────────────────────────────────────────
p_psd = gw.addPlot(title="PSD — Welch (blink-cleaned, 1 Hz HPF)")
p_psd.setLabel('left',   'PSD (µV²/Hz)')
p_psd.setLabel('bottom', 'Frequency (Hz)')
p_psd.setXRange(0, 50)
p_psd.setYRange(0, 1)
p_psd.showGrid(x=True, y=True, alpha=0.2)
for band, (lo, hi) in BANDS.items():
    p_psd.addItem(pg.LinearRegionItem(
        [lo, hi], brush=pg.mkBrush(BAND_COLORS[band]+'22'), movable=False))
curve_psd = p_psd.plot(pen=pg.mkPen('#A56EE5', width=1.5))

p_bar = gw.addPlot(title="Relative Band Power (3.5–30 Hz normalised)")
p_bar.setLabel('left', 'Relative Power')
p_bar.setYRange(0, 1)
p_bar.showGrid(y=True, alpha=0.2)
p_bar.getAxis('bottom').setTicks(
    [[(i, n) for i, n in enumerate(BANDS.keys())]])

bar_items = []
for i, (name, color) in enumerate(zip(BANDS.keys(), BAND_COLORS.values())):
    b = pg.BarGraphItem(x=[i], height=[0], width=0.6,
                        brush=pg.mkBrush(color),
                        pen=pg.mkPen('w', width=0.5))
    p_bar.addItem(b)
    bar_items.append(b)

gw.nextRow()

# ── Row 3: ML panel ──────────────────────────────────────────────────────────
p_ml = gw.addPlot(title="ML Classification Panel", colspan=2)
p_ml.hideAxis('bottom')
p_ml.hideAxis('left')
p_ml.setXRange(0, 10)
p_ml.setYRange(0, 4)
p_ml.setFixedHeight(100)

# ML-1
ml1_hdr = pg.TextItem("Focus Classifier (ML-1):", color='#F4A261', anchor=(0,1))
ml1_hdr.setFont(QtGui.QFont('Arial', 10))
p_ml.addItem(ml1_hdr); ml1_hdr.setPos(0.3, 3.8)

ml1_result = pg.TextItem("Load a model with --focus-model", color='#888888', anchor=(0,1))
# FIX (PyQt6): QFont.Bold -> QFont.Weight.Bold
ml1_result.setFont(QtGui.QFont('Arial', 13, QtGui.QFont.Weight.Bold))
p_ml.addItem(ml1_result); ml1_result.setPos(0.3, 2.7)

# ML-2
ml2_hdr = pg.TextItem("N-Band Classifier (ML-2):", color='#7B68EE', anchor=(0,1))
ml2_hdr.setFont(QtGui.QFont('Arial', 10))
p_ml.addItem(ml2_hdr); ml2_hdr.setPos(5.2, 3.8)

ml2_result = pg.TextItem("Load a model with --nband-model", color='#888888', anchor=(0,1))
ml2_result.setFont(QtGui.QFont('Arial', 13, QtGui.QFont.Weight.Bold))
p_ml.addItem(ml2_result); ml2_result.setPos(5.2, 2.7)

main_win.show()

# =============================================================================
# STATE
# =============================================================================
sample_count   = 0
current_state  = "Calibrating…"
_psd_ylim_max  = 1.0
_ml1_label, _ml1_score = ("No model loaded", 0.0)
_ml2_label, _ml2_conf  = ("No model loaded", 0.0)

# =============================================================================
# UPDATE LOOP
# =============================================================================
def update():
    global sample_count, current_state, _psd_ylim_max
    global _ml1_label, _ml1_score, _ml2_label, _ml2_conf

    ser = get_serial()
    if ser is None:
        state_label.setText("State: Serial disconnected — retrying…")
        return

    for _ in range(10):
        # FIX: don't even attempt a read if nothing is buffered yet —
        # avoids repeated near-blocking calls when the ESP32 is idle/disconnected.
        if ser.in_waiting == 0:
            break
        try:
            raw_line = ser.readline().decode('utf-8', errors='ignore')
        except serial.SerialException:
            with ser_lock:
                global _ser
                _ser = None
            return

        val, hw_ts = parse_line(raw_line)
        if val is None:
            continue

        sample_count += 1
        drift_mon.tick(hw_ts)

        filtered = filter_sample_display(float(val))
        raw_buf.append(float(val))
        filt_buf.append(filtered)

        epoch_buf.append(val)
        if len(epoch_buf) == epoch_buf.maxlen:
            w = np.array(epoch_buf, dtype=float)
            if float(np.max(np.abs(w - np.mean(w)))) > ARTIFACT_THRESHOLD:
                print(f"[ARTIFACT] sample {sample_count}")

        if sample_count % PSD_UPDATE_INTERVAL == 0:
            filt_array = np.array(filt_buf)

            # DEBUG: print live signal stats every ~2.5s so we can see what's
            # actually arriving, instead of guessing why calibration stalls.
            if sample_count % (PSD_UPDATE_INTERVAL * 25) == 0:
                raw_arr = np.array(raw_buf)
                print(f"[DEBUG] raw ADC min={int(np.min(raw_arr))} "
                      f"max={int(np.max(raw_arr))} | "
                      f"filt std={np.std(filt_array):.2f} | "
                      f"baseline: {len(baseline.alpha_buf)}/{baseline.target} "
                      f"ready={baseline.ready}")

            cleaned, _ = blink_detector.clean(
                filt_array, sample_offset=sample_count - BUFFER_SIZE)
            blink_clean_buf.extend(cleaned.tolist())
            ml_epoch_acc.extend(cleaned[-PSD_UPDATE_INTERVAL:].tolist())

            # ML inference
            if len(ml_epoch_acc) >= ML_EPOCH_SAMPLES:
                epoch_arr = np.array(list(ml_epoch_acc), dtype=float)
                ratio_feats, abs_powers = extract_ml_features(epoch_arr, FS)
                _ml1_label, _ml1_score  = predict_focus(ratio_feats)
                _ml2_label, _ml2_conf   = predict_nband(abs_powers)

            freqs, psd = compute_psd_clean(blink_clean_buf)
            if freqs is not None:
                powers = compute_powers(freqs, psd)
                if powers is not None:
                    baseline.update(powers['alpha_n'], powers['beta_n'])
                    current_state = baseline.classify(powers)

                    for bar, key in zip(bar_items, BANDS.keys()):
                        bar.setOpts(height=[powers[key+'_n']])

                    mask = freqs <= 50
                    f50, p50 = freqs[mask], psd[mask]
                    curve_psd.setData(f50, p50)
                    new_max = float(np.max(p50)) * 1.15 + 1e-9
                    if abs(new_max - _psd_ylim_max)/(_psd_ylim_max+1e-9) > 0.2:
                        p_psd.setYRange(0, new_max)
                        _psd_ylim_max = new_max

                    # FIX: Slice the actual time-series segment and pass exactly 6 args
                    latest_chunk = list(filt_buf)[-PSD_UPDATE_INTERVAL:]
                    send_to_matlab(current_state, powers, latest_chunk,
                                   blink_detector.blink_count,
                                   _ml1_label, _ml2_label)

                    log_row(current_state, powers,
                            blink_detector.blink_count,
                            _ml1_label, _ml1_score,
                            _ml2_label, _ml2_conf)

    # Optimization: Reuse the array cast from the loop logic block above
    try:
        curve_filt.setData(np.array(filt_buf))
    except Exception:
        pass

    # Update state/blink labels
    state_label.setText(f"State: {current_state}")
    state_label.setPos(0, -y_scale * 0.85)
    blink_label.setText(f"Blinks: {blink_detector.blink_count}")
    blink_label.setPos(BUFFER_SIZE, -y_scale * 0.85)

    # Update ML panel
    if _focus_interpreter is not None:
        fc = '#2ECC71' if 'High' in _ml1_label else '#E74C3C'
        ml1_result.setText(f"{_ml1_label}  ({_ml1_score:.0%})")
        ml1_result.setColor(fc)
    else:
        ml1_result.setText("No model — pass --focus-model path")
        ml1_result.setColor('#666666')

    if _nband_model is not None:
        ml2_result.setText(f"{_ml2_label}  ({_ml2_conf:.0%})")
        ml2_result.setColor('#7B68EE')
    else:
        ml2_result.setText("No model — pass --nband-model path")
        ml2_result.setColor('#666666')

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(16)

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("  EEG Monitor V5 (Fixed)")
    print(f"  Port: {PORT} | Baud: {BAUD} | ADC: {BITS}-bit | FS: {FS} Hz")
    print(f"  Focus model : {args.focus_model or 'NOT LOADED — pass --focus-model'}")
    print(f"  N-Band model: {args.nband_model or 'NOT LOADED — pass --nband-model'}")
    print(f"  Log file    : {log_path}")
    print("  Drag the Y-Scale slider to zoom in/out on the waveform.")
    print("  Close window or Ctrl+C to stop.")
    print("=" * 70)
    try:
        # FIX (PyQt6): exec_() was removed; use exec()
        sys.exit(app.exec())
    except KeyboardInterrupt:
        pass
    finally:
        _csv_file.close()
        if _ser and _ser.is_open:
            _ser.close()
        if udp_sock:
            udp_sock.close()
        print(f"\n[SAVED] {log_path}")
        print("[DONE]")
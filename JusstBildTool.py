import csv
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

APP_NAME = "Jusst BildTool"
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))
else:
    APP_DIR = Path(__file__).resolve().parent
    BUNDLE_DIR = APP_DIR

TOOLS_DIR = APP_DIR / "tools"
ESR_DIR = TOOLS_DIR / "realesrgan"
ESR_EXE = ESR_DIR / "realesrgan-ncnn-vulkan.exe"
ESR_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def metrics(img):
    arr = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    means = arr.reshape(-1, 3).mean(axis=0)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "rgb_means": means,
    }


def gray_world(img):
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    means = arr.reshape(-1, 3).mean(axis=0)
    avg = float(means.mean())
    gains = np.clip(avg / np.maximum(means, 1.0), 0.85, 1.15)
    arr *= gains
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def auto_correct(img, strong=False):
    changed = []
    m = metrics(img)

    if m["brightness"] < 85:
        img = ImageEnhance.Brightness(img).enhance(1.20 if strong else 1.12)
        changed.append("Belichtung+")
    elif m["brightness"] > 205:
        img = ImageEnhance.Brightness(img).enhance(0.90 if strong else 0.95)
        changed.append("Belichtung-")

    if m["contrast"] < 42:
        img = ImageOps.autocontrast(img, cutoff=0.5)
        changed.append("Kontrast")

    means = m["rgb_means"]
    ratio = float(np.max(means) / max(np.min(means), 1.0))
    if ratio > (1.07 if strong else 1.09):
        img = gray_world(img)
        changed.append("Weissabgleich")

    m2 = metrics(img)
    if m2["sharpness"] < (130 if strong else 100):
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115 if strong else 80, threshold=3))
        changed.append("Schaerfe")

    return img, changed


def remove_red_eye(img):
    rgb = np.asarray(img.convert("RGB")).copy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.12, 5, minSize=(60, 60))
    changed = False

    for fx, fy, fw, fh in faces:
        upper = gray[fy:fy + int(fh * 0.65), fx:fx + fw]
        eyes = eye_cascade.detectMultiScale(upper, 1.10, 6, minSize=(15, 10))
        for ex, ey, ew, eh in eyes:
            x, y = fx + ex, fy + ey
            roi = rgb[y:y + eh, x:x + ew]
            if roi.size == 0:
                continue
            r = roi[:, :, 0].astype(np.int16)
            g = roi[:, :, 1].astype(np.int16)
            b = roi[:, :, 2].astype(np.int16)
            yy, xx = np.ogrid[:eh, :ew]
            cx, cy = ew / 2.0, eh / 2.0
            central = ((xx - cx) ** 2 / (0.48 * ew + 1) ** 2 + (yy - cy) ** 2 / (0.55 * eh + 1) ** 2) <= 1
            mask = (r > 90) & (r > g * 1.35 + 10) & (r > b * 1.35 + 10) & central
            if mask.sum() < 3:
                continue
            replacement = ((g + b) / 2).astype(np.uint8)
            roi[:, :, 0][mask] = replacement[mask]
            changed = True

    return Image.fromarray(rgb), changed


def resize_long_edge(img, target):
    if target <= 0:
        return img, False
    w, h = img.size
    edge = max(w, h)
    if edge == target:
        return img, False
    scale = target / edge
    resized = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
    return resized, True


def ensure_realesrgan(status_callback=None):
    if ESR_EXE.exists():
        return True
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    ESR_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = TOOLS_DIR / "realesrgan.zip"
    try:
        if status_callback:
            status_callback("Real-ESRGAN wird einmalig geladen …")
        urllib.request.urlretrieve(ESR_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(ESR_DIR)
        zip_path.unlink(missing_ok=True)
        found = list(ESR_DIR.rglob("realesrgan-ncnn-vulkan.exe"))
        if not found:
            return False
        exe = found[0]
        if exe.parent != ESR_DIR:
            for item in exe.parent.iterdir():
                target = ESR_DIR / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), str(target))
        return ESR_EXE.exists()
    except Exception:
        zip_path.unlink(missing_ok=True)
        return False


def ai_upscale(img, temp_dir, status_callback=None):
    if not ensure_realesrgan(status_callback):
        return img, False, "KI nicht verfügbar; Lanczos"
    inp = temp_dir / "input.png"
    out = temp_dir / "output.png"
    out.unlink(missing_ok=True)
    img.save(inp)
    try:
        proc = subprocess.run(
            [str(ESR_EXE), "-i", str(inp), "-o", str(out), "-n", "realesrgan-x4plus", "-s", "4"],
            cwd=str(ESR_DIR), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900,
        )
        if proc.returncode == 0 and out.exists():
            with Image.open(out) as im:
                return im.convert("RGB").copy(), True, ""
    except Exception:
        pass
    return img, False, "KI nicht verfügbar; Lanczos"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("880x670")
        self.minsize(780, 580)
        self.files = []
        self.output_dir = None

        ttk.Label(self, text=APP_NAME, font=("Segoe UI", 20, "bold")).pack(pady=(12, 4))
        ttk.Label(self, text="JPG/JPEG → PNG · Metadaten löschen · Auto-Korrektur · Rote Augen · KI-Upscaling").pack()

        buttons = ttk.Frame(self)
        buttons.pack(pady=10)
        ttk.Button(buttons, text="Bilder auswählen", command=self.choose_files).pack(side="left", padx=4)
        ttk.Button(buttons, text="Ordner auswählen", command=self.choose_folder).pack(side="left", padx=4)
        ttk.Button(buttons, text="Ausgabeordner", command=self.choose_output).pack(side="left", padx=4)
        ttk.Button(buttons, text="Liste löschen", command=self.clear).pack(side="left", padx=4)

        self.listbox = tk.Listbox(self, height=15)
        self.listbox.pack(fill="both", expand=True, padx=12)

        frame = ttk.Frame(self)
        frame.pack(fill="x", padx=12, pady=10)

        ttk.Label(frame, text="Profil").grid(row=0, column=0, sticky="w")
        self.profile = tk.StringVar(value="AUTO")
        ttk.Combobox(frame, textvariable=self.profile, values=["NORMAL", "AUTO", "MAX"], state="readonly", width=12).grid(row=0, column=1, padx=5)

        ttk.Label(frame, text="Zielkante").grid(row=0, column=2, sticky="e")
        self.size = tk.StringVar(value="3840")
        ttk.Combobox(frame, textvariable=self.size, values=["0", "1920", "2560", "3000", "3840", "4096"], width=10).grid(row=0, column=3, padx=5)
        ttk.Label(frame, text="px").grid(row=0, column=4, sticky="w")

        self.red = tk.BooleanVar(value=True)
        self.ai = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Rote Augen automatisch", variable=self.red).grid(row=1, column=0, columnspan=2, pady=8, sticky="w")
        ttk.Checkbutton(frame, text="KI-Upscaling wenn nötig", variable=self.ai).grid(row=1, column=2, columnspan=3, sticky="w")

        self.output_label = ttk.Label(self, text="Ausgabe: automatisch neben den ausgewählten Bildern")
        self.output_label.pack(pady=(0, 4))
        self.status = ttk.Label(self, text="Bereit")
        self.status.pack(pady=5)
        self.progress = ttk.Progressbar(self)
        self.progress.pack(fill="x", padx=12)
        ttk.Button(self, text="ALLE BILDER VERARBEITEN", command=self.process_all).pack(fill="x", padx=12, pady=12, ipady=8)

    def set_status(self, text):
        self.status.config(text=text)
        self.update_idletasks()

    def choose_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Bilder", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")])
        for f in files:
            self.add(Path(f))

    def choose_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        for f in Path(folder).rglob("*"):
            if f.suffix.lower() in SUPPORTED:
                self.add(f)

    def choose_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_dir = Path(folder)
            self.output_label.config(text=f"Ausgabe: {self.output_dir}")

    def add(self, p):
        p = p.resolve()
        if p not in self.files:
            self.files.append(p)
            self.listbox.insert("end", str(p))

    def clear(self):
        self.files.clear()
        self.listbox.delete(0, "end")

    def process_all(self):
        if not self.files:
            messagebox.showwarning("Keine Bilder", "Bitte zuerst Bilder oder einen Ordner auswählen.")
            return

        target = int(self.size.get())
        base = self.output_dir or (self.files[0].parent / "JusstBildTool_Fertig")
        output = Path(base)
        changed_dir = output / "Geaendert"
        temp = output / "_temp"
        output.mkdir(parents=True, exist_ok=True)
        changed_dir.mkdir(exist_ok=True)
        temp.mkdir(exist_ok=True)

        logs = []
        total = len(self.files)
        self.progress["maximum"] = total
        self.progress["value"] = 0

        for i, src in enumerate(self.files, 1):
            self.set_status(f"{i}/{total}: {src.name}")
            notes = []
            try:
                with Image.open(src) as im:
                    img = ImageOps.exif_transpose(im).convert("RGB")

                if self.red.get():
                    img, did = remove_red_eye(img)
                    if did:
                        notes.append("Rote Augen")

                if self.profile.get() != "NORMAL":
                    img, n = auto_correct(img, self.profile.get() == "MAX")
                    notes += n

                edge = max(img.size)
                need_ai = False
                if self.ai.get() and target > 0:
                    if self.profile.get() == "MAX":
                        need_ai = edge < target
                    elif self.profile.get() == "AUTO":
                        need_ai = edge < target * 0.70

                if need_ai:
                    img, did, fallback_note = ai_upscale(img, temp, self.set_status)
                    if did:
                        notes.append("KI-Upscaling")
                    elif fallback_note:
                        notes.append(fallback_note)

                img, resized = resize_long_edge(img, target)
                if resized:
                    notes.append("Skalierung")

                out = output / f"{src.stem}.png"
                count = 1
                while out.exists():
                    out = output / f"{src.stem}_{count}.png"
                    count += 1

                img.save(out, "PNG", optimize=True)

                if notes:
                    shutil.copy2(out, changed_dir / out.name)

                logs.append([str(src), str(out), "; ".join(notes)])
            except Exception as exc:
                logs.append([str(src), "FEHLER", str(exc)])

            self.progress["value"] = i
            self.update_idletasks()

        with open(output / "Verarbeitung.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Original", "Ausgabe", "Aenderungen"])
            writer.writerows(logs)

        shutil.rmtree(temp, ignore_errors=True)
        self.set_status(f"Fertig: {output}")
        try:
            os.startfile(output)
        except Exception:
            pass
        messagebox.showinfo("Fertig", f"Verarbeitung abgeschlossen.\n\n{output}")


if __name__ == "__main__":
    App().mainloop()

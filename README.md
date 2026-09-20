# Jusst BildTool

Eigenständiges Windows-Tool zur Stapelverarbeitung von Bildern.

Funktionen:
- JPG/JPEG/PNG/BMP/TIFF/WebP einlesen
- Ausgabe als frische PNG-Dateien ohne übernommene EXIF-/GPS-/IPTC-/XMP-Metadaten
- automatische Helligkeits-, Kontrast-, Weißabgleich- und Schärfekorrektur
- konservative Rote-Augen-Korrektur
- optionales KI-Upscaling mit Real-ESRGAN
- Profile NORMAL / AUTO / MAX
- Zielkante bis 4096 px
- CSV-Protokoll
- Originaldateien bleiben unverändert

## Windows-Build

Unter **Actions → Build Windows EXE** wird eine eigenständige `JusstBildTool.exe` erzeugt und als Artifact `JusstBildTool-Windows` bereitgestellt.

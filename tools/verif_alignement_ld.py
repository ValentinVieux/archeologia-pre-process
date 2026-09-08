"""Contrôle d'alignement entre deux rasters censés représenter le même terrain (ex. LD
livré vs LD régénéré, LD vs MNT IGN) par corrélation de phase sur une grille de fenêtres
DANS chaque dalle kilométrique testée. Diagnostic (leçon 2026-09-06, dalles 2 201 px
géoréférencées comme 1 000 m : décalage nul au centre, ±50 m aux bords) :

  - décalages ~0 partout                    -> ALIGNÉ
  - décalage constant                       -> TRANSLATION (origine fausse)
  - décalage qui change de signe N/S ou E/W -> ÉCHELLE (taille de pixel / couverture fausse)

Verdict CONFORME si toutes les fenêtres exploitables sont sous --tolerance m. Sans GPU.

    .venv/Scripts/python.exe tools/verif_alignement_ld.py <raster_ref> <raster_test> [--dalles N] [--fenetre 300] [--tolerance 1.0]
"""
import argparse
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

RES = 0.5  # grille d'analyse (m)


def lire(src, b, cote):
    n = int(round(cote / RES))
    fill = src.nodata if src.nodata is not None else 0
    return src.read(1, window=from_bounds(*b, transform=src.transform), out_shape=(n, n),
                    resampling=Resampling.bilinear, boundless=True, fill_value=fill).astype("float32"), fill


def phase(a, b):
    """Décalage (dx, dy) en pixels de b par rapport à a, et netteté du pic."""
    a = a - a.mean(); b = b - b.mean()
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    A, B = np.fft.fft2(a * win), np.fft.fft2(b * win)
    R = A * np.conj(B); R /= np.abs(R) + 1e-9
    c = np.fft.ifft2(R).real
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    dy = iy if iy <= a.shape[0] // 2 else iy - a.shape[0]
    dx = ix if ix <= a.shape[1] // 2 else ix - a.shape[1]
    return dx, dy, c.max() / (np.abs(c).mean() + 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raster_ref", help="référence (ex. LD régénéré depuis le MNT IGN)")
    ap.add_argument("raster_test", help="raster à contrôler (ex. LD livré)")
    ap.add_argument("--dalles", type=int, default=6, help="nombre de dalles km testées (réparties sur l'emprise commune)")
    ap.add_argument("--fenetre", type=float, default=300.0, help="côté des fenêtres de corrélation (m)")
    ap.add_argument("--tolerance", type=float, default=1.0, help="décalage toléré (m)")
    a = ap.parse_args()
    with rasterio.open(a.raster_ref) as R, rasterio.open(a.raster_test) as T:
        x0, y0 = max(R.bounds.left, T.bounds.left), max(R.bounds.bottom, T.bounds.bottom)
        x1, y1 = min(R.bounds.right, T.bounds.right), min(R.bounds.top, T.bounds.top)
        kx0, ky0, kx1, ky1 = int(np.ceil(x0 / 1000)), int(np.ceil(y0 / 1000)), int(x1 // 1000) - 1, int(y1 // 1000) - 1
        cand = [(kx, ky) for kx in range(kx0, kx1 + 1) for ky in range(ky0, ky1 + 1)]
        rng = np.random.default_rng(0)
        rng.shuffle(cand)
        print(f"emprise commune {x0:.0f} {y0:.0f} {x1:.0f} {y1:.0f} ; ref {R.res[0]:.4f} m/px, test {T.res[0]:.4f} m/px")
        mesures, testees = [], 0
        for kx, ky in cand:
            if testees >= a.dalles:
                break
            X0, Y0 = kx * 1000.0, ky * 1000.0
            lignes, ok = [], 0
            for fy in (850, 500, 150):
                cellules = []
                for fx in (150, 500, 850):
                    cx, cy = X0 + fx, Y0 + fy
                    b = (cx - a.fenetre / 2, cy - a.fenetre / 2, cx + a.fenetre / 2, cy + a.fenetre / 2)
                    ra, fr = lire(R, b, a.fenetre); ta, ft = lire(T, b, a.fenetre)
                    if (ra != fr).mean() < 0.95 or (ta != ft).mean() < 0.95 or ra.std() < 2 or ta.std() < 2:
                        cellules.append("    vide    "); continue
                    dx, dy, pic = phase(ra, ta)
                    mesures.append((dx * RES, -dy * RES, fx, fy)); ok += 1
                    cellules.append(f"{dx * RES:+5.1f},{-dy * RES:+5.1f} m")
                lignes.append(f"   y+{fy:3d} : " + " | ".join(cellules))
            if ok >= 3:
                testees += 1
                print(f"dalle {kx:04d}_{ky:04d} :"); print("\n".join(lignes))
        if not mesures:
            print("NON CONFORME : aucune fenêtre exploitable"); sys.exit(1)
        m = np.array(mesures)
        dxy = np.hypot(m[:, 0], m[:, 1])
        # signature d'échelle : corrélation entre position dans la dalle et décalage
        cx = np.corrcoef(m[:, 2], m[:, 0])[0, 1] if m[:, 0].std() > 0.3 else 0.0
        cy = np.corrcoef(m[:, 3], m[:, 1])[0, 1] if m[:, 1].std() > 0.3 else 0.0
        print(f"\n{len(m)} fenêtres sur {testees} dalles : décalage médian ({np.median(m[:, 0]):+.1f}, {np.median(m[:, 1]):+.1f}) m, "
              f"max {dxy.max():.1f} m ; corrélation position/décalage x {cx:+.2f}, y {cy:+.2f}")
        if dxy.max() <= a.tolerance:
            print("CONFORME : rasters alignés"); return
        diag = "ÉCHELLE (taille de pixel ou couverture de dalle fausse)" if max(abs(cx), abs(cy)) > 0.45 else \
               ("TRANSLATION (origine fausse)" if dxy.std() < a.tolerance else "DÉCALAGE VARIABLE (à inspecter)")
        print(f"NON CONFORME : {diag}"); sys.exit(1)


if __name__ == "__main__":
    main()

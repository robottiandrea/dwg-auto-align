# DWG Auto Align MVP (Windows)

Tool Python per allineare automaticamente una tavola DWG non georiferita a un DWG rilievo georiferito, stimando una trasformazione di similarità 2D (scala uniforme + rotazione + traslazione).

## Struttura progetto

- `main.py` - entry point CLI/GUI (Tkinter), orchestrazione completa
- `cad_io.py` - conversione temporanea DWG<->DXF con ODA File Converter
- `point_extractor.py` - estrazione entità `POINT` in XY
- `matcher.py` - matching robusto e RANSAC
- `transform.py` - modello matematico e stima LS
- `apply_transform.py` - applicazione trasformazione a tutte le entità
- `report.py` - report JSON + TXT
- `config.py` - configurazione centrale
- `requirements.txt` - dipendenze Python minime

## Prerequisiti

1. **Windows 10/11**
2. **Python 3.10+**
3. **ODA File Converter** installato
4. File input in formato **DWG** contenenti entità `POINT`

## Installazione dipendenze

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configurazione ODA File Converter

Di default il codice usa:

`C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe`

Puoi configurare in due modi:

1. Variabile ambiente:

```bash
set ODA_FILE_CONVERTER=C:\path\to\ODAFileConverter.exe
```

2. File `config.local.json` nella root progetto:

```json
{
  "oda_converter_path": "C:\\Program Files\\ODA\\ODAFileConverter\\ODAFileConverter.exe",
  "matching_tolerance": 0.5,
  "ransac_max_iterations": 2500,
  "min_confidence": 0.55,
  "low_confidence_warn_only": true
}
```

## Avvio programma

### Modalità GUI (consigliata)

```bash
python main.py
```

Campi:
- DWG rilievo
- DWG tavola progetto
- cartella output
- tolleranza

Pulsanti:
- **Analizza**
- **Allinea e salva**

### Modalità CLI

```bash
python main.py --nogui --survey C:\data\rilievo.dwg --project C:\data\progetto.dwg --out C:\data\out --tolerance 0.5 --force-save
```

## Output generati

Nella cartella output:

1. `*_aligned.dwg` (mai sovrascrive originale)
2. `*_aligned_report.json`
3. `*_aligned_report.txt`
4. log esecuzione (`dwg_auto_align.log`, nella cartella output o temp)

## Algoritmo MVP

1. Conversione interna temporanea DWG->DXF (invisibile all'utente)
2. Estrazione `POINT` XY da rilievo/progetto
3. RANSAC su coppie minime (2+2 punti) per ipotesi di similarità
4. Valutazione inlier con nearest-neighbor entro tolleranza e vincolo di unicità
5. Refinement con least squares su inlier
6. Calcolo RMS e confidenza
7. Applicazione trasformazione a tutte le entità del DWG progetto
8. Conversione finale DXF->DWG

## Modalità demo/validazione

Esegue un test sintetico con parametri noti (scala/rotazione/traslazione) e rumore moderato:

```bash
python main.py --demo
```

Stampa confronto tra parametri reali e stimati, numero inlier, RMS e confidenza.

## Limiti attuali

- MVP assume differenze solo di **similarità 2D** (no affine, no deformazioni locali).
- Dipendenza da ODA File Converter installato e raggiungibile.
- Entità complesse (es. alcune varianti `DIMENSION`) possono non essere trasformabili da `ezdxf.transform` in tutti i casi: vengono loggate come non supportate.
- Matching basato su `POINT`; se i `POINT` sono pochi o molto ambigui la confidenza può essere bassa.

## Estensioni future

- Miglioramento punteggio di confidenza con metriche statistiche avanzate
- Spatial indexing (KD-tree) per dataset molto grandi
- Supporto multi-layout/paperspace
- Export report HTML/PDF
- Batch processing di più coppie DWG

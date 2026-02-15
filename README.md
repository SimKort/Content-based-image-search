# Content-based-image-search (Turiniu pagrįsta batų vaizdų paieška su CLIP + Streamlit)

## Apžvalga
Šis projektas įgyvendina turiniu pagrįstą vaizdų paiešką batų nuotraukoms.
Vartotojas įkelia užklausos (query) bato nuotrauką, pasirenka kiek rezultatų grąžinti (Top-K) ir kategoriją (moteriški ir vyriški / moteriški / vyriški). Sistema grąžina N panašiausių batų nuotraukų su jų pavadinimais pagal vaizdo turinį.

Projektas naudoja iš anksto apmokytą **OpenAI CLIP** modelį vaizdų savybių žymėms (embedding’ams) išgauti ir **cosine** atstumą panašumui skaičiuoti. Modelio mokymas nuo nulio nevykdomas.

## Metodai
### 1) Savybių žymės (embeddings) su CLIP
- Naudojamas CLIP `ViT-B/32` modelis.
- Kiekvienam katalogo vaizdui iš anksto sugeneruojamas vektorius (embedding) ir išsaugomas `data/clip_features.npz`.
- Užklausos nuotrauka taip pat užkoduojama į embedding.

### 2) Panašumo paieška
- Vaizdų požymių vektoriai normuojami pagal formulę:
  
  $x_i^{(normuota)}=\frac{x_i}{\sqrt{\sum_{j=1}^{n}x_j^2}}$.
- Artimiausių vaizdų paieška atliekama pagal `sklearn.neighbors.NearestNeighbors(metric="cosine")`, kurios metu skaičiuojamas panašumas tarp skirtingų vektorių pagal kosinusą ir grąžinama $n$ artimiausių kaimynų.
- Rezultatai rikiuojami pagal mažiausią cosine atstumą (didžiausią panašumą `sim = 1 - distance`).

### 3) UI (Streamlit)
- Failo įkėlimas (query image).
- Top-K slankiklis (kiek rezultatų rodyti).
- Kategorijos filtras (moteriški ir vyriški/moteriški/vyriški).
- Rezultatų atvaizdavimas.
- Po paieškos sugeneruojamas CSV su rezultatais į `results/`.

## Repo struktūra
```text
.streamlit/
  config.toml
data/
  examples/
  clip_features.npz
  products_with_subcategory.csv
notebooks/
  nd1.ipynb
results/
  last_results.csv
src/
  ui/
    app_streamlit.py
  __init__.py
README.md
requirements.txt
```

## Duomenys
- `data/products_with_subcategory.csv` – metaduomenys (ID;Name;Category;Description;Price;Image URLs;Subcategory).
- `data/clip_features.npz` – iš anksto sugeneruoti CLIP embeddings:
  - `paths` – originalūs vaizdų keliai / ID šaltinis
  - `features` – embedding vektoriai

> Pastaba: rezultatai UI atvaizduojami pagal `Image URLs` lauką iš CSV.

## Įdiegimo instrukcijos
### 1) Gaukite programos failus iš GitHub
```text
git clone https://github.com/SimKort/Content-based-image-search.git
cd Content-based-image-search
```

### 2) Sukurkite virtualią aplinką
```text
cd Content-based-image-search
python -m venv .venv
```

### 3) Aktyvuokite .venv
`.\.venv\Scripts\Activate.ps1`
> Jei PowerShell neleidžia vykdyti skriptų:

`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 4) Įdiekite priklausomybes
```text
python -m pip install -U pip
pip install -r requirements.txt
```

### 5) Įdiekite CLIP
```text
pip install "setuptools<82"
pip install --no-build-isolation git+https://github.com/openai/CLIP.git
```

### 6) Paleidimas
`streamlit run src/ui/app_streamlit.py`
> Atsidarys naršyklėje (pvz. http://localhost:8501). Pirmas užkrovimas ir užklausos apdorojimas užtrunka daugiau laiko, po to viskas vykdoma labai greitai.

## Naudojimas
- Įkelkite bato nuotrauką (PNG/JPG/JPEG).
- Pasirinkite kiek panašių batų rodyti.
- Pasirinkite kategoriją: Moteriški ir vyriški / Moteriški / Vyriški.
- Peržiūrėkite gautus rezultatus.
- Po paieškos automatiškai išsaugomas rezultatų CSV į results/.

## Žinomos problemos / apribojimai
- CLIP diegimas Windows + Python 3.12: gali reikėti setuptools<82 ir --no-build-isolation (žr. sąranką).
- Duomenų priklausomybė nuo URL: jei kai kurie Image URLs nepasiekiami (404), tuomet dalis rezultatų gali nerodyti nuotraukos.
- Kategorijų neatitikimai: filtras remiasi Category reikšmėmis CSV. Jei duomenyse yra skirtingas žymėjimas, filtravimą reikia adaptuoti.
- Veikimo greitis: embeddings naudojami iš npz, bet užklausos embedding skaičiuojamas realiu laiku. CPU režime tai gali būti lėčiau.

## Naudotos technologijos
- Python, NumPy, Pandas
- PyTorch
- OpenAI CLIP (ViT-B/32)
- scikit-learn (NearestNeighbors)
- Streamlit (UI)
- PIL (Pillow) vaizdų apdorojimui

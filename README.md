# Content-based-image-search (Turiniu pagrįsta batų vaizdų paieška su CLIP + Streamlit)

## Apžvalga
Šis projektas įgyvendina turiniu pagrįstą vaizdų paiešką batų nuotraukoms.
Vartotojas įkelia užklausos (query) bato nuotrauką, pasirenka kiek rezultatų grąžinti ir kategoriją (moteriški ir vyriški / moteriški / vyriški). Sistema grąžina N panašiausių batų nuotraukų su jų pavadinimais pagal vaizdo turinį.

Projekte naudojamas iš anksto apmokytas **OpenAI CLIP** modelis vaizdų požymių vektoriams išgauti ir **cosine** atstumą panašumui skaičiuoti. Modelio mokymas nuo nulio nevykdomas.

## Metodai
### 1) Vaizdų požymių vektoriai su CLIP
- Naudojamas CLIP `ViT-B/32` modelis.
- Kiekvienam katalogo vaizdui iš anksto sugeneruojamas vektorius ir išsaugomas `data/clip_features.npz`.
- Taikant šį modelį užklausos nuotrauka taip pat užkoduojama į vektorių.

### 2) Panašumo paieška
- Vaizdų požymių vektoriai normuojami pagal formulę:
  
$x_i^{(normuota)}=\frac{x_i}{\sqrt{\sum_{j=1}^{n}x_j^2}}$.

- Panašių vaizdų paieškai sudaromas modelis `sklearn.neighbors.NearestNeighbors`. Taikant `kneighbors` pagal kosinusą palyginamas ieškomo vaizdo vektorius su kitų vaizdų vektoriais ir grąžinami atstumai bei top pasirinktų panašiausių vaizdų indeksai.

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
Duomenys buvo paimti iš eavalyne.lt svetainės. Jie buvo surinkti automatiniu būdu (žr. `notebooks/nd1.ipynb`). Iš viso ištraukta 6451 skirtingų batų, tarp kurių 3322 moteriški ir 3129 vyriški. Tarp **moteriškų** batų buvo gautos šios kategorijos: aukštakulniai, auliniai batai, balerinos, batai uždaroms aikštelėms, batai vandens sportui, batai į sporto salę, aulinukai, bateliai, botfortai, bėgimo batai, guminiai batai, ilgaauliai, jojikų batai, kaubojiški batai, kedai, kerzai, laisvalaikio batai, loaferai, lordsai, mokasinai, naminės šlepetės, oksfordo batai, pusbačiai, sniego batai, sportbačiai, teniso batai, turistiniai batai, štibletai, žygio batai. Tarp **vyriškų** batų gautos šios kategorijos: auliniai batai, aulinukai, batai uždaroms aikštelėms, batai vandens sportui, batai į sporto salę, bokso batai, bėgimo batai, futbolo batai, guminiai batai, ilgaauliai, kedai, krepšinio batai, laisvalaikio batai, loaferai, lordsai, mokasinai, naminės šlepetės, pusbačiai, sniego batai, sportbačiai, teniso batai, turistiniai batai, štibletai, žygio batai. Kiek kiekvienoje kategorijoje batų žr. `notebooks/nd1.ipynb`.

**Duomenų struktūra:**
- `data/products_with_subcategory.csv` – metaduomenys (ID; Name; Category; Description; Price; Image URLs; Subcategory).
- `data/clip_features.npz` – iš anksto sugeneruoti CLIP požymių vektoriai:
  - `paths` – vaizdų vietos.
  - `features` – požymių vektoriai.
- `data/images` - visi ištraukti vaizdai, su kuriais atliekamas palyginimas.

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
- CLIP diegimas Windows + Python 3.12: gali reikėti setuptools<82 ir --no-build-isolation (žr. įdiegimo instrukcijos).
- Duomenų priklausomybė nuo URL: jei kai kurie nuotraukų URLs nepasiekiami (404), tuomet dalis rezultatų gali nerodyti nuotraukos.
- Veikimo greitis: požymių vektoriai naudojami iš `data/clip_features.npz`, bet užklausos embedding skaičiuojamas realiu laiku. CPU režime tai gali būti lėčiau.
- Kadangi kai kuriose kategorijose yra mažai vaizdų, daliai batų gali būti pateikti netikslūs rezultatai.
- Kadangi modelis yra bendrinis, tai gauti požymių vektoriai gali nepakankamai informatyviai atspindėti vaizdą.

## Naudotos technologijos
- Python, NumPy, Pandas
- PyTorch
- OpenAI CLIP (ViT-B/32)
- scikit-learn (NearestNeighbors, cosine_similarity)
- Streamlit (UI)
- PIL (Pillow) vaizdų apdorojimui
- BeautifulSoup, requests, urllib, re, csv, os, shutil, time – duomenų paruošimas
- Matplotlib – vizualizacija

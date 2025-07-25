# AI Health Assistant

# Via Predictiva (ML) — AI Health Assistant

**Predicció del cansament a partir de dades fisiològiques de wearables**

Aquest repositori conté la **via predictiva** del projecte *AI Health Assistant*. Aquí es desenvolupa, entrena i avalua el model de *machine learning* encarregat d’estimar el risc de **cansament** d’un usuari a partir de mesures fisiològiques diàries (HR, HRV, son, activitat, etc.).

> 💡 En l’arquitectura final del projecte, el model predictiu funciona com un **indicador de risc sensible**, complementant el mòdul interpretatiu (LLM), que és qui tradueix les dades i el risc en recomanacions accionables.

---

## Objectius

1. **Construir un pipeline reproduïble** de preprocessament, enginyeria de característiques i reequilibri de classes.
2. **Entrenar i optimitzar** diversos models de classificació i seleccionar el millor d’acord amb mètriques orientades a minimitzar *falsos negatius* (maximitzar *recall* de la classe *cansat*).
3. **Avaluar rigorosament** el rendiment en un conjunt de prova mantingut, amb informes i gràfiques de suport.
4. **Exportar el model** per a la seva integració a la *web app*.

---

## Dades

* **Conjunt de dades:** *LifeSnaps* (71 participants, mesures diàries).
* **Etiqueta objectiu:** percepció de **cansament** (fatiga subjectiva) el dia següent.
* **Granularitat:** registre/dia per usuari; s’afegeixen agregacions temporals i derivades fisiològiques.

> ⚠️ El dataset és desequilibrat: menys casos «cansat» que «no cansat». Al pipeline es fa servir **SMOTETomek** per reequilibrar durant l’entrenament.

---

## Metodologia

### 1) Preprocessament & EDA

* Neteja de valors perduts i outliers.
* Normalització/estandardització segons model.
* *Feature engineering* de mètriques HRV (RMSSD), qualitat del son, càrrega d’activitat i tendències.
* Partició **train/test** per evitar *leakage*.

### 2) Reequilibri de classes

* Aplicació de **SMOTETomek** només sobre *train*, mantenint *test* intactes.

### 3) Models provats

* Baselines: **Logistic Regression**, **Random Forest**, **XGBoost**.
* Model final: **LightGBM (LGBM)** amb ajust d’hiperparàmetres i **validació creuada**.

### 4) Criteri d’optimització

* Priorització del **Recall** de la classe *cansat* i **F1-score** per reduir *falsos negatius*.

### 5) Reporting

* Matrius de confusió, corbes d’aprenentatge, importàncies de característiques i *permutation importance*.

---

## Resultats

Rendiment del **model final (LGBM)** al conjunt de prova:

| Mètrica               | Valor     |
| --------------------- | --------- |
| **F1-Score**          | **0,61**  |
| **Recall (cansat)**   | **77,3%** |
| **Precisió (cansat)** | **50,6%** |
| **Exactitud global**  | **62,0%** |
| **Falsos positius**   | **133**   |
| **Falsos negatius**   | **40**    |

**Interpretació.** El model és **sensible** detectant episodis reals de fatiga, però amb una precisió moderada; produeix falses alarmes. És adequat com a **indicador de risc**, no com a «jutge» per a decisions crítiques. Per això, a la solució final es combina amb el mòdul LLM, que genera el pla d’acció.

---

## 📁 Estructura del Repositori

```
📁 AI-Health-Assistant  
├── 📁 .venv                     # Entorn virtual de Python per a la gestió d'entorns i dependències  
├── 📁 .vscode                   # Configuracions específiques de Visual Studio Code  
├── 📁 data                     # Dataset brut i processat emprat per entrenar i validar models  
├── 📁 llm                      # Exportació de les dades d'entrenament de LifeSnaps per ajustar el LLM  
├── 📁 models                   # Models entrenats
├── 📁 notebooks                # Notebooks de Jupyter per a EDA, proves i experiments
├── 📁 results                  # Resultats generats com gràfics, mètriques i logs d’experiments  
│   ├── 📁 01_EDA\ figures        # Figures i gràfics de l’anàlisi exploratòria de dades  
│   ├── 📁 02_experiments         # Resultats dels experiments amb models de ML  
│   └── 📁 03_training            # Outputs relacionats amb el procés d'entrenament (matrius de confusió i corves d'aprenentatge)
├── 📁 src                      # Codi font del projecte  
│   └── 📁 ai_health_assistant   # Mòdul principal del projecte amb la lògica organitzada per fases  
│       ├── 📁 01_preprocessing    # Funcions i scripts per a la preparació i neteja de dades  
│       ├── 📁 02_training         # Entrenament del model de predicció de fatiga  
│       ├── 📁 03_assistant        # Mòdul de proves de predicció  
│       └── 📁 utils               # Funcions auxiliars i utilitats comunes  
│       └── 📄 __init__.py         # Inicialització del paquet Python  
├── 📄 .env                     # Variables d’entorn (paths, credencials, etc.)  
├── 📄 .gitattributes           # Configuracions Git (per a GitHub)  
├── 📄 .gitignore               # Llista d’arxius i carpetes que Git ha d’ignorar  
├── 📄 README.md                # Descripció general del projecte i instruccions d’ús  
├── 📄 requirements-py313.txt   # Llista de dependències per a Python 3.13  
└── 📄 setup.py                 # Script per instal·lar el paquet com a llibreria Python  
```

---

## Instal·lació

> Requereix **Python 3.13**.

```bash
# 1) Clonar el repo
git clone https://github.com/RogerDuran808/AI-Health-Assistant.git
cd AI-Health-Assistant

# 2) Crear entorn virtual
python -m venv .venv

# 3) Activar-lo
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 4) Instal·lar dependències i el paquet en mode editable
pip install -r requirements-py313.txt
pip install -e .
```


---

### Nomenclatura de versions

- v0.1.0 → Projecte / notebook en proves
    - v0.1.1 → Correcció d'algun error o petites modificacions
- v1.0.0 → Primera versión completa del notebook o projecte
    - v1.1.0 → Funcions noves del projecte o notebook

---

## Limitacions i línies futures

* El senyal de **fatiga subjectiva d’un sol dia** presenta límits conceptuals; futurs models haurien d’incorporar **finestra temporal multidiària**, context d’entrenament i factors psicosocials.

* Limitació de la granularitat de les dades: només un registre/dia per usuari i pocs usuaris per tenir una representació realista de la població.

---

## Citacions

Si uses aquest treball, cita el projecte *Assistent de Salut basat en IA* i el conjunt de dades *LifeSnaps* corresponent.

---

## Contacte

**Autor:** Roger Duran López
**Tutor acadèmic:** Guillem Guigó i Corominas

Per a dubtes tècnics o col·laboracions, obre una *issue* o envia’m un correu.








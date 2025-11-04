# SC4020 – Data Mining Project
## Task 1: Symptom Co-occurrence (Apriori)

### Folder layout (relevant parts)
```
.
├─ data/
│  ├─ raw/                      
│  │  ├─ dataset.csv
│  │  ├─ symptom_Description.csv
│  │  ├─ symptom_precaution.csv
│  │  └─ Symptom-severity.csv
│  └─ interim/                 
├─ outputs/
│  └─ task1/                   
├─ src/
│  ├─ task1_apriori/
│  │  ├─ prep.py              
│  │  └─ apriori.py            
│  └─ utils/
│     ├─ text_normalize.py
│     └─ symptom_synonyms.json
├─ requirements.txt
└─ README.md
```

---

### 1) Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run the pipeline

#### a) Build transactions
```bash
python3 -m src.task1_apriori.prep   --input data/raw/dataset.csv   --output data/interim/transactions.csv   --synonyms src/utils/symptom_synonyms.json
```
- Detects `Symptom_*` columns
- Normalizes tokens (underscores & synonyms)
- De-duplicates within each disease
- Writes `disease_id,symptoms` (semicolon-delimited list)

#### b) Mine frequent itemsets & rules
```bash
python3 -m src.task1_apriori.apriori   --transactions data/interim/transactions.csv   --min_support 0.10   --min_conf 0.60   --max_len 3   --min_transactions 200   --outdir outputs/task1
```

### 3) Outputs (in `outputs/task1/`)
- `frequent_itemsets.csv` – itemsets with support  
- `rules.csv` – association rules (antecedents, consequents, support, confidence, lift)  
- `rules_top_lift.csv` – top-K by lift (default 50)  
- `rules_top_confidence.csv` – top-K by confidence (default 50)  
- `support_vs_conf.png` – scatter of rule support vs confidence  
- `lift_hist.png` – histogram of rule lift  
- `rules_graph.png` – directed graph of top-lift rules


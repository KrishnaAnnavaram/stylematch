# data/

Git ignores every file in this folder except this README. Do not commit the catalog, ratings or labels.
The offline demo and the tests do not need this folder: `stylematch synth --out data` writes a synthetic
catalog and a labelled query set with the same columns.

## Expected files

| File | Columns | Used by |
|---|---|---|
| `catalog.csv` | `ProductID`, `ProductName`, `ProductBrand`, `Gender`, `Price (INR)`, `Description`, `PrimaryColor` (snake_case names `product_id`, `name`, `brand`, `gender`, `price`, `description`, `colour` also work). Required: ID, name, gender, price | `index`, `recommend`, `evaluate`, `pool`, `study-plan` |
| `gold.jsonl` | One JSON object per line: `query_id`, `query`, `relevant` (`{"<product_id>": grade}`, grade 1 or 2) | `evaluate` |
| `labels.csv` (alternative gold set) | `query_id`, `query`, `product_id`, `relevance` (0, 1 or 2) | `evaluate` |
| `queries.jsonl` | `query_id`, `query` | `pool`, `study-plan` |

## Sources

| Data | Source | URL | License / terms |
|---|---|---|---|
| Fashion product catalog (about 12,500 products, Indian e-commerce site) | Kaggle "Fashion Products Catalog" dataset (`Fashion_products_catalog.csv`) | https://www.kaggle.com/datasets (search "fashion products catalog myntra") | Read the Kaggle dataset license before you use or share it. Do not commit it. |
| Relevance labels | Your own annotators, with `stylematch pool` | — | Your own data |
| User-study ratings | Your own participants, with `stylematch study-plan` | — | Get consent. Store ratings without names. |

The descriptions of the public catalog contain brand helpline numbers and e-mail addresses.
The loader removes them before indexing (`catalog.scrub_contacts`).

## How to prepare the real data

1. Download `Fashion_products_catalog.csv` from Kaggle and save it as `data/catalog.csv`.
2. Run `stylematch index`. The first run builds the index in `artifacts/index`.
3. Write `data/queries.jsonl` with the queries that you want to label.
4. Run `stylematch pool --queries data/queries.jsonl --out data/pool.csv`.
5. Give the pool to annotators. They fill `relevance` with 0, 1 or 2 without knowing which system found the product.
6. Run `stylematch evaluate --gold data/pool.csv`.

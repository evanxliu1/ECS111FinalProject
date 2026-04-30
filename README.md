# Cross-Generator AI Review Detector — ECS111 Final Project

Building a binary classifier (human vs AI) for product reviews that **generalizes across LLM generators**, using contrastive loss + a generator-adversarial head.

## Setup (Google Colab — free T4)

```python
!git clone <your-repo-url> /content/ECS111FinalProject
%cd /content/ECS111FinalProject
!pip install -q -r requirements.txt
```

Mount Drive and put API keys in a `.env` file:
```
OPENAI_API_KEY=...
HF_TOKEN=...
```

## End-to-end run order

1. **Generate data** — `notebooks/01_generate_data.ipynb`
   - Loads 10k pre-2022 Amazon reviews (human pool).
   - Generates 2.5k AI reviews from each of: GPT-5 mini, Granite 4.1 8B, Gemma-4 E4B-it, Qwen3.6 (small).
2. **Build splits** — `notebooks/02_build_splits.ipynb`
   - Produces 4 leave-one-generator-out (LOGO) rotations as parquet files.
3. **Train all variants** — `notebooks/03_train_all.ipynb`
   - 4 rotations × 4 model variants (base, contrastive, adversarial, both) = 16 runs.
4. **Evaluate** — `notebooks/04_evaluate.ipynb`
   - Aggregates results into the in-dist vs cross-gen comparison table.

## Project layout

See `src/` for the implementation. Configs in `configs/` toggle the loss components.

## Local smoke tests

```bash
python -m src.data.load_human --limit 100
python -m src.data.generate_openai --n 5
python -m src.data.build_splits --rotation 0 --debug
python -m src.train --config configs/base.yaml --rotation 0 --max-steps 50 --debug
```

# Adversial Financial NLP Vulnerability Analysis

Adversial financial NLP vulnerability analysis and improvement using LLMs.

## Usage

```bash
git clone https://github.com/gitmichaelqiu/AdvFinNLPVuln.git
cd AdvFinNLPVuln
pip install -r requirements.txt
```

Rename `.env.example` to `.env` and configure it.

```bash
python run main.py
```

## License

This project is licensed under MIT License. See [LICENSE](./LICENSE) for details.

## Acknowledgement

- The `financial_news_dataset.csv` dataset used in this project is licensed under Apache 2.0. See [LICENSE](./licenses/LICENSE-financial-news-dataset.txt) for details. The dataset is available at [Kaggle](https://www.kaggle.com/datasets/mikemiller125/kaggleyahoo-finance-news).
    - This dataset is derived from [financial-news-dataset](https://github.com/FelixDrinkall/financial-news-dataset) by FelixDrinkall under CC BY-NC-SA 4.0 License.
- The `kaggle_fake_news_FULL.csv` dataset used in this project is licensed under CC0 1.0. See [LICENSE](./licenses/LICENSE-financial-news-classification-dataset.txt) for details. The dataset is available at [Kaggle](https://www.kaggle.com/datasets/mikemiller125/financial-news-classification-dataset).

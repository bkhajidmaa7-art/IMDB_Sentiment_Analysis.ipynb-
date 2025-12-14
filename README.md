# IMDB_Sentiment_Analysis.ipynb-
# IMDB Сэтгэл хөдлөлийн Анализ

**Dataset:** IMDB Large Movie Review Dataset  
[Dataset-ийг татах холбоос](https://ai.stanford.edu/~amaas/data/sentiment/)

**Төслийн зорилго:**  
- Кино шүүмжийг эерэг/сөрөг гэж ангилах  
- TF-IDF, Word2Vec, BERT embedding-үүдийг турших  
- Машин сургалтын алгоритмууд (Logistic Regression, Random Forest, AdaBoost) хоорондын гүйцэтгэлийг харьцуулах

---

## 1. Даалгаврын тайлбар
Энэхүү төсөл нь IMDB кино шүүмжийг **positive / negative** гэж ангилах зорилготой.  
Олон embedding арга, олон машин сургалтын алгоритмыг туршиж хамгийн сайн хослолыг тодорхойлно.

---

## 2. Dataset
- IMDB Large Movie Review Dataset (~80MB)  
- Train: 25,000 review, Test: 25,000 review  
- Label: `positive` / `negative`  
- Dataset **repo-д ороогүй**, хэмжээ их тул

**Препроцессинг арга:**
- Бүх үсгийг жижиг үсэг болгох  
- Тэмдэгт устгах  
- Tokenize хийх  
- Stopwords устгах (TF-IDF, Word2Vec)  
- BERT-д зориулж padding хийх

---

## 3. Feature Embeddings
| Арга | Дэлгэрэнгүй |
|------|------------|
| **TF-IDF** | Max features = 5000, unigram & bigram |
| **Word2Vec** | CBOW & Skip-gram, vector size = 300 |
| **BERT** | `bert-base-uncased`, `roberta-base`, `sbert`, `albert`, `hatebert` ашигласан |

---

## 4. Машин сургалтын моделүүд
| Модел | Hyperparameters | Тайлбар |
|-------|----------------|---------|
| Logistic Regression | solver=`liblinear`, C=1.0 | Суурь загвар |
| Random Forest | n_estimators=100, max_depth=10 | Модны ансамбль |
| AdaBoost | n_estimators=100, learning_rate=1.0 | Boosting арга |

> Бүх модель train/test split (25k/25k) дээр сурсан. Cross-validation хэрэглэх боломжтой.

---

## 5. Туршилтын орчин
- Environment: Google Colab / Ubuntu 22.04  
- Python хувилбар: 3.10  
- Library: `torch`, `transformers`, `scikit-learn`, `pandas`  
- Runtime: CPU (BERT Colab runtime дээр сурсан)  
- Туршилт бүрийг **3 удаа давтсан**  
- Hyperparameter tuning гар аргаар хийгдсэн

---

## 6. Туршилтын үр дүн

| Embedding | Модел | Accuracy | F1-score |
|-----------|-------|---------|----------|
| TF-IDF    | Logistic Regression | 0.8794 | 0.8794 |
| TF-IDF    | AdaBoost            | 0.7524 | 0.7522 |
| TF-IDF    | Random Forest       | 0.8429 | 0.8428 |
| Word2Vec  | Logistic Regression | 0.8097 | 0.8097 |
| Word2Vec  | AdaBoost            | 0.7602 | 0.7602 |
| Word2Vec  | Random Forest       | 0.7702 | 0.7702 |
| BERT      | Logistic Regression | …      | …      |
| BERT      | Random Forest       | …      | …      |
| BERT      | AdaBoost            | …      | …      |

> Хамгийн сайн гүйцэтгэл: **TF-IDF + Logistic Regression**  
> BERT үр дүн runtime-ийн хязгаарлалттай учир хараахан гараагүй

---

## 7. Дүгнэлт
- TF-IDF + Logistic Regression хамгийн сайн гүйцэтгэлтэй  
- Word2Vec embedding арай муу, BERT хэрвээ бүрэн сурвал илүү сайн байх магадлалтай  
- Ensemble моделүүд (Random Forest, AdaBoost) robustness сайжруулдаг, accuracy бага зэрэг нэмдэг  
- Ирээдүйд: BERT fine-tuning, илүү advanced архитектур турших

---

## 8. Ашигласан материал, сангууд
1. Maas, Andrew L., et al. “Learning Word Vectors for Sentiment Analysis.” *ACL 2011.*  
2. Devlin, Jacob, et al. “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.” *NAACL 2019.*  
3. Mikolov, Tomas, et al. “Efficient Estimation of Word Representations in Vector Space.” *arXiv 2013.*  
… (өөрийн ашигласан paper, blog, github link-үүдийг нэм)

---

## 9. Багшийн анхаарах зүйл
- Notebook-д **бүх код**, preprocessing, embedding, моделийн сургалт, evaluation багтсан  
- GitHub preview **Colab metadata.widgets issue**-ээс болж эвдэрсэн байж болно  
- Notebook **Google Colab / Jupyter дээр зөв ажиллана**


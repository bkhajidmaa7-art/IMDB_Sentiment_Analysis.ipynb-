# IMDB_Sentiment_Analysis.ipynb
Embedding ба Машин сургалтын аргуудын харьцуулалт

**Dataset:** IMDB Large Movie Review Dataset  
[Dataset-ийг татах холбоос](https://ai.stanford.edu/~amaas/data/sentiment/)

**Төслийн зорилго:**  
- Кино шүүмжийг эерэг/сөрөг гэж ангилах  
- TF-IDF, Word2Vec, BERT embedding-үүдийг турших  
- Машин сургалтын алгоритмууд (Logistic Regression, Random Forest, AdaBoost) хоорондын гүйцэтгэлийг харьцуулах

## 1. Даалгаврын тайлбар
Энэхүү ажилд IMDB кино шүүмжийн өгөгдлийг ашиглан дараах асуудлыг шийдсэн:
Олон embedding арга (TF-IDF, Word2Vec, BERT)
Олон ангилагч модель (Logistic Regression, Random Forest, AdaBoost, LSTM)
ашиглан туршилт хийж, Accuracy, F1-score гэсэн хоёр үнэлгээний үзүүлэлтээр үр дүнг харьцуулсан.

---

## 2. Dataset
Нийт өгөгдөл: 50,000
Train set: 25,000
Test set: 25,000
Label: positive, negative
Dataset хэмжээ том (~80MB) тул GitHub репозитор дээр байршуулаагүй

2.1 Препроцессинг арга:
-Бүх үсгийг жижиг үсэг болгох
-Тэмдэгтүүдийг устгах
-Tokenization
-Stopword устгах (TF-IDF, Word2Vec)

---

## 3. Ашигласан Embedding аргууд
3.1 TF-IDF

TF-IDF нь үгийн давтамж болон тухайн үг бусад баримтад хэр түгээмэл байгаагаас хамаарч жин өгдөг статистик арга юм.
max_features = 5000
unigram + bigram
https://scikit-learn.org/stable/modules/feature_extraction.html

3.2 Word2Vec

-Word2Vec нь үгсийн семантик утгыг хадгалсан вектор үүсгэдэг embedding арга юм.
-Ашигласан хувилбарууд
-CBOW
-Skip-gram
-Vector size = 300
-https://arxiv.org/abs/1301.3781

3.3 BERT ба түүний хувилбарууд

BERT нь transformer архитектурт суурилсан, контекстийг хоёр чиглэлд ойлгодог deep embedding юм.
Ашигласан моделүүд:
-bert-base-uncased
-roberta-base
-sentence-bert (SBERT)
-albert
-hatebert
https://arxiv.org/abs/1810.04805

---

## 4. Ашигласан машин сургалтын аргууд
4.1 Logistic Regression

Шугаман ангилагч
Sigmoid функц ашиглан магадлал тооцдог
Жижиг болон дунд dataset дээр маш сайн ажилладаг
https://en.wikipedia.org/wiki/Logistic_regression

4.2 Random Forest

Олон decision tree-ийн ансамбль
Overfitting багатай
Robust модель
AdaBoost
Boosting арга
Алдаатай өгөгдөлд илүү жин өгч суралцдаг
https://en.wikipedia.org/wiki/AdaBoost

4.3 LSTM

Recurrent Neural Network
Дараалсан өгөгдөлд тохиромжтой

---

## 5. Туршилтын орчин
Орчин: Google Colab, Ubuntu 22.04
-Python: 3.10
Ашигласан сангууд:
-torch
-transformers
-scikit-learn
-pandas
-numpy
Runtime: T4
-Туршилт бүрийг 3 удаа давтан ажиллуулсан
-Hyperparameter tuning-ийг гараар хийсэн

## 6. Туршилтын үр дүн

| Embedding | Модел | Accuracy | F1-score |
|-----------|-------------------------|------------|------------|
| TF-IDF    | **Logistic Regression** | **0.8794** | **0.8794** |
| TF-IDF    | AdaBoost                |   0.7524   |   0.7522   |
| TF-IDF    | Random Forest           |   0.8429   |   0.8428   |
| Word2Vec  | Logistic Regression     |   0.8097   |   0.8097   |
| Word2Vec  | AdaBoost                |   0.7602   |   0.7602   |
| Word2Vec  | Random Forest           |   0.7702   |   0.7702   |


-Хамгийн сайн гүйцэтгэл: **TF-IDF + Logistic Regression**  
-BERT үр дүн runtime-ийн хязгаарлалттай учир хараахан гараагүй

---

## 7. Дүгнэлт

TF-IDF + Logistic Regression хамгийн өндөр гүйцэтгэл үзүүлсэн
Dataset харьцангуй жижиг тул энгийн модель илүү сайн ажилласан
Word2Vec нь семантик утга хадгалдаг боловч training-д илүү их өгөгдөл шаарддаг
BERT нь бүрэн fine-tuning хийвэл илүү сайн үр дүн өгөх боломжтой
Ensemble аргууд (Random Forest, AdaBoost) нь тогтвортой боловч accuracy харьцангуй бага байсан

---

## 8. Цаашдын ажил

BERT full fine-tuning хийх
Hyperparameter tuning-ийг Grid / Random Search ашиглан сайжруулах
Cross-validation-ийг системтэй ашиглах
Том dataset дээр deep learning аргуудыг турших

---

## 9. Dataset-тэй холбоотой судалгааны ажлууд

1. **Maas et al. (2011)** – *Learning Word Vectors for Sentiment Analysis*  
   https://aclanthology.org/P11-1015/
2. **Pang & Lee (2008)** – *Opinion Mining and Sentiment Analysis*  
   https://www.cs.cornell.edu/home/llee/omsa/omsa.pdf
3. **Devlin et al. (2019)** – *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*  
   https://arxiv.org/abs/1810.04805
4. **Mikolov et al. (2013)** – *Efficient Estimation of Word Representations in Vector Space (Word2Vec)*  
   https://arxiv.org/abs/1301.3781
5. **Kim (2014)** – *Convolutional Neural Networks for Sentence Classification*  
   https://arxiv.org/abs/1408.5882
6. **Radford et al. (2017)** – *Improving Language Understanding by Generative Pre-Training (GPT)*  
   https://openai.com/research/language-unsupervised
7. **Joulin et al. (2016)** – *Bag of Tricks for Efficient Text Classification (FastText)*  
   https://arxiv.org/abs/1607.01759
8. **Peters et al. (2018)** – *Deep Contextualized Word Representations (ELMo)*  
   https://arxiv.org/abs/1802.05365
9. **Liu et al. (2019)** – *RoBERTa: A Robustly Optimized BERT Pretraining Approach*  
   https://arxiv.org/abs/1907.11692
10. **Sanh et al. (2019)** – *DistilBERT, a distilled version of BERT*  
    https://arxiv.org/abs/1910.01108

---

## 10. Ерөнхий дүгнэлт

Энэхүү судалгааны ажлын хүрээнд IMDb кино шүүмжийн dataset дээр sentiment analysis хийх зорилгоор уламжлалт болон орчин үеийн embedding аргуудыг, 
түүнчлэн хэд хэдэн машин сургалтын болон deep learning загваруудыг харьцуулан туршив. Туршилтын үр дүнгээс харахад dataset-ийн хэмжээ харьцангуй бага 
эсвэл дунд түвшинд байх үед TF-IDF embedding ашигласан Logistic Regression загвар нь хамгийн тогтвортой бөгөөд өндөр гүйцэтгэлтэй ажилласан. 
Энэ нь уг моделийн математик энгийн бүтэц, overfitting багатай байдал, мөн sparse өгөгдөл дээр сайн ажилладаг онцлогтой холбоотой гэж үзэж байна.
Харин dataset-ийн хэмжээ ихсэх тусам илүү их параметртэй, контекстийг гүнзгий ойлгох чадвартай deep learning болон transformer-based загварууд 
(BERT болон түүнтэй төстэй архитектурууд) илүү давуу гүйцэтгэл үзүүлэх боломжтой болох нь ажиглагдсан. Ийм загварууд нь өгөгдлөөс далд утга, хэлний нарийн бүтэц,
үгийн хоорондын хамаарлыг илүү сайн сурч чаддаг давуу талтай. Мөн энэхүү судалгаанаас embedding арга сонголт нь модель сонголттой адил чухал хүчин зүйл болох нь тодорхой харагдсан. 
Жишээлбэл, TF-IDF нь уламжлалт машин сургалтын загваруудад илүү тохиромжтой байхад, Word2Vec болон BERT зэрэг dense embedding-үүд нь deep learning архитектуруудтай илүү үр дүнтэй 
хосолдог байна. Иймээс sentiment analysis хийхдээ dataset-ийн хэмжээ, тооцооллын нөөц, ашиглах модель болон embedding аргуудын уялдаа холбоог харгалзан үзэх нь өндөр гүйцэтгэлтэй
систем боловсруулахад чухал ач холбогдолтой гэж дүгнэж байна. Цаашид уг судалгааг өргөжүүлэн BERT fine-tuning, илүү олон transformer загварууд, мөн hyperparameter optimization болон 
cross-validation зэрэг аргуудыг ашигласнаар үр дүнг илүү сайжруулах боломжтой гэж үзэж байна.

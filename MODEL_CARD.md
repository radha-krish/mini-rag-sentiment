# Sentiment Analysis Model Card

## 1. Dataset Overview

**Dataset splits:**

| Split | Samples |
|-------|---------|
| Train | 224     |
| Dev   | 48      |
| Test  | 48      |

**Label distribution:**

- Train: 0 → 115, 1 → 109  
- Dev: 0 → 23, 1 → 25  
- Test: 0 → 23, 1 → 25  

**Characteristics:**

- Average text length: ~132 characters (~18 words)  
- Total emojis in training: 15  
- No missing values  
- Small, roughly balanced dataset with repetitive product mentions  

---

## 2. Baseline Approach (Experiment 1)

**Preprocessing:** Lowercasing, URL removal, emoji → text, punctuation cleaning  
**Feature Extraction:** TF-IDF (max_features=5000)  
**Model:** Logistic Regression (max_iter=1000)  

**Dev Set Metrics:**

| Metric    | Label 0 | Label 1 | Macro Avg | Weighted Avg |
|-----------|---------|---------|-----------|--------------|
| Precision | 0.581   | 0.706   | 0.643     | 0.646        |
| Recall    | 0.783   | 0.480   | 0.631     | 0.625        |
| F1-score  | 0.667   | 0.571   | 0.619     | 0.617        |
| Accuracy  | 0.625 |

**Observations:**  

- Negative reviews are better predicted than positive.  
- TF-IDF captures repeated tokens but not context.  
- Small dataset limits performance.  

---

## 3. Enhanced Preprocessing Baseline (Experiment 2)

**Enhancements:** SpaCy tokenization, modular `clean_text` function, optional lemmatization/stopword removal, emoji preservation  
**Feature Extraction:** TF-IDF (same as Experiment 1)  
**Model:** Logistic Regression (max_iter=10000)  

**Results:** Accuracy = 0.625 (no improvement yet)  

**Key Takeaways:**  

- Modular preprocessing pipeline is scalable and ready for embeddings or transformer models.  
- TF-IDF alone cannot capture semantic meaning.  

---

## 4. Multi-Model Comparison (Experiment 3)

| Model             | Accuracy | Macro F1 | Observations |
|------------------|----------|----------|--------------|
| Logistic Regression | 0.625   | 0.619    | Baseline linear model |
| Linear SVM        | 0.667   | 0.666    | Better positive class recall |
| Random Forest     | 0.688   | 0.684    | Highest accuracy; handles repetitive tokens |
| Gradient Boosting | 0.646   | 0.644    | Balanced, slightly lower accuracy |

**Observations:**  

- Random Forest performs best on small, repetitive datasets.  
- Preprocessing improvements benefit all models.  

---

## 5. Embeddings + Random Forest (Experiment 5)

**Embedding:** all-MiniLM-L6-v2 (384-dim)  
**Classifier:** Random Forest (50 trees)  
**Preprocessing:** Lowercasing, emoji → text  

**Results:**  

- Train accuracy: 1.0 → overfitting  
- Dev accuracy: 0.5833  
- Biased towards negative class  

**Analysis:** Random Forest struggles with high-dimensional embeddings on small datasets; poor generalization.  

---

## 6. Embeddings + Linear SVM (Experiment 7)

**Classifier:** Linear SVM (C=1.0, class_weight='balanced', max_iter=5000)  

**Results:**  

- Train accuracy: 0.9598  
- Dev accuracy: 0.75  
- Balanced precision and recall across classes  

**Insights:**  

- SVM generalizes better than Random Forest with embeddings.  
- Class balancing improves minority class recall.  

---

## 7. Transformer Fine-Tuning (Experiment 8)

**Model:** DistilBertForSequenceClassification  
**Tokenizer:** DistilBertTokenizerFast  
**Training:** 5 epochs, batch size 16, max token length 128, learning rate 2e-5  

**Training / Dev Accuracy:**

| Epoch | Train Acc | Dev Acc |
|-------|-----------|---------|
| 1     | 0.6205    | 0.5208  |
| 2     | 0.9152    | 0.8958  |
| 3     | 0.9866    | 0.9375  |
| 4     | 1.0000    | 0.9167  |
| 5     | 1.0000    | 0.9375  |

**Final Dev Set Metrics:**

| Class | Precision | Recall  | F1-score |
|-------|-----------|---------|----------|
| 0     | 0.917     | 0.957   | 0.936    |
| 1     | 0.958     | 0.920   | 0.939    |

**Observations:**  

- Rapid learning; minor overfitting at Epoch 4  
- Handles both classes effectively  
- Best performance across all experiments (Dev F1 ~0.94)  

---

## 8. Comparative Summary

| Experiment | Model / Features       | Dev Accuracy | Key Takeaway |
|------------|----------------------|-------------|--------------|
| 1          | TF-IDF + LR           | 0.625       | Baseline; limited semantic capture |
| 2          | Enhanced TF-IDF + LR  | 0.625       | Robust preprocessing; future-ready |
| 3          | TF-IDF, multiple models | 0.688 (RF) | RF handles repetitive small datasets well |
| 5          | MiniLM + RF           | 0.583       | Overfits embeddings; poor generalization |
| 7          | MiniLM + Linear SVM   | 0.75        | Good generalization; embeddings effective |
| 8          | DistilBERT fine-tune  | 0.9375      | Best performance; robust class-wise F1 |

---

## 9. Error Analysis & Robustness

**Baseline:** Misclassified positive reviews; fails on semantic nuances.  
**Embeddings + RF:** Overfitting; biased towards negative class.  
**Embeddings + SVM:** Balanced but misses some subtle sentiment cues.  
**DistilBERT:** Minor errors on neutral/sarcastic text; robust overall.  

**Robustness Observations:**  

- Handles emojis after conversion to text  
- Pre-trained transformer weights mitigate small dataset limitations  
- Modular preprocessing allows easy future feature additions  

---

## 10. Conclusions & Recommendations

1. **Best Approach:** DistilBERT fine-tuning achieves highest accuracy (93.75%) and balanced F1.  
2. **Preprocessing:** Modular pipeline from Experiment 2 is scalable.  
3. **Embedding-based Models:** Linear SVM preferred over Random Forest for small datasets.  
4. **Future Work:** Data augmentation, hyperparameter tuning, ensemble methods, and production deployment.  

✅ **Final Remark:**  
Fine-tuning DistilBERT provides robust sentiment classification on small datasets, outperforming TF-IDF baselines and embedding + classical ML models, ready for deployment.

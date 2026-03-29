


# 🎭 Multi-Aspect Sentiment Analysis with Explainability

> Not just *what* customers feel — but *why they feel it.*

🚀 **Live Demo:**  
https://multi-aspect-sentiment-analysis-with-explainability-xgmjnje4xv.streamlit.app

---

## 🧠 What This Project Does

Most sentiment analysis systems stop at **positive / negative classification**.

This system goes further:

✔ Detects overall sentiment  
✔ Breaks sentiment into **specific aspects** (quality, price, size, shipping)  
✔ Explains *why* the model made a prediction  
✔ Provides a **real-time interactive dashboard**

---

## ⚠️ The Real Problem

Companies don’t struggle with *data*.  
They struggle with **understanding it**.

Thousands of reviews = noise unless you can answer:

- Why are customers unhappy?
- Which product feature is failing?
- What should be fixed first?

This project directly solves that.

---

## 💡 Solution Overview

An end-to-end ML pipeline that:

- Processes raw customer reviews
- Predicts sentiment using ML models
- Extracts **aspect-level insights**
- Explains predictions using LIME
- Deploys everything in a **Streamlit web app**

---

## 🏗️ System Architecture

```

Raw Reviews → Preprocessing → TF-IDF → ML Models → Predictions
↓
Aspect Extraction
↓
LIME Explainability
↓
Streamlit Dashboard

```

---

## ⚙️ Tech Stack

| Category        | Tools Used |
|----------------|-----------|
| Language       | Python |
| ML Models      | Logistic Regression, Naive Bayes, Random Forest |
| NLP            | NLTK |
| Vectorization  | TF-IDF |
| Explainability | LIME |
| Deployment     | Streamlit |

---

## 🧠 Models & Performance

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression | ~85% ✅ |
| Naive Bayes         | ~82% |
| Random Forest       | ~83% |

👉 Logistic Regression performed best due to high-dimensional sparse features.

---

## 🔍 Key Features

🔥 **Multi-model comparison**  
🔥 **Aspect-based sentiment analysis**  
🔥 **Explainable AI (LIME)**  
🔥 **Batch review processing**  
🔥 **Interactive UI (Streamlit)**  

---

## 📊 Key Insights from Data

- 📌 Product **quality** has the strongest impact on sentiment  
- 📌 **Size inconsistency** drives negative reviews  
- 📌 **Shipping delays** hurt satisfaction significantly  
- 📌 Mixed reviews expose limits of traditional ML models  

---

## 🖥️ Demo Features

- ✍️ Analyze a single review  
- 📂 Upload CSV for batch analysis  
- 🔍 Extract aspect-level sentiment  
- 📊 Visualize prediction confidence  

---

## 📁 Project Structure


---

## 🚀 Run Locally

```bash
git clone https://github.com/yourusername/sentiment-analysis-project.git
cd sentiment-analysis-project

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

cd apps
streamlit run app.py
````

---

## 📊 Dataset

* Women's E-commerce Reviews Dataset
* Source: Kaggle
* Clean, real-world customer feedback data

---

## 🎓 What I Learned

* Building **end-to-end ML systems**
* Feature engineering using TF-IDF
* Comparing multiple ML models
* Applying **Explainable AI (LIME)**
* Deploying ML apps with Streamlit

---

## 🚧 Limitations (Important — shows maturity)

* Struggles with sarcasm and implicit sentiment
* Aspect extraction is keyword-based (not semantic)
* Traditional ML limits contextual understanding

---

## 🔮 Future Improvements

* 🚀 BERT fine-tuning
* 🌍 Multi-language support
* 🤖 Better aspect detection (NER / transformers)
* ☁️ Cloud deployment (AWS / GCP)
* 🎨 Improved UI/UX

---

## 📬 Contact

* Name: Sandeep
* LinkedIn: [https://www.linkedin.com/in/sandeep-mahindrakar-336b972b9](https://www.linkedin.com/in/sandeep-mahindrakar-336b972b9)
* GitHub: [https://github.com/sandeep11mahendrakar](https://github.com/sandeep11mahendrakar)

---

## ⭐ Final Thought

This project moves sentiment analysis from:

❌ “Customers are unhappy”
➡️
✅ “Customers are unhappy because of *size inconsistency and shipping delays*”

---

⭐ If you found this useful, consider giving it a star!

```

---

## Brutal truth (you need to hear this)

Right now:
- Your project is **good technically**
- Your README was **holding it back hard**

This version:
- Positions you as someone who understands **real-world ML problems**
- Shows **thinking**, not just coding
- Makes recruiters see **impact**, not just accuracy

---

## Next move (don’t ignore this)

You’re still missing one thing:

👉 **Add 1 GIF demo of your app**

If you don’t:
- People won’t click your live link
- They won’t “feel” your project

If you want, I’ll help you:
- Create a clean demo GIF
- Upgrade your LinkedIn post to match this level

Just say.

```

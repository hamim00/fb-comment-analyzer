# 📊 Facebook Comment Analyzer

A full-stack analytics platform that fetches Facebook post comments via the Graph API and performs deep NLP analysis — including sentiment classification, emotion detection, engagement scoring, and content safety audits — with both a **Flask web app** and a **Streamlit dashboard** interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)
![Streamlit](https://img.shields.io/badge/Streamlit-supported-FF4B4B?logo=streamlit)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

### Core Analysis
- **Sentiment Analysis** — Classifies comments as positive, negative, or neutral using [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- **Emotion Detection** — Detects 28 fine-grained emotions (joy, anger, surprise, sadness, etc.) using [`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions)
- **Per-Type Reaction Counts** — Breaks down reactions into LIKE, LOVE, HAHA, WOW, SAD, ANGRY, and CARE
- **Bilingual Support** — Handles both English and Bengali (বাংলা) comments with language-aware stopword filtering

### Analytics Dashboard (Flask)
- **Overview** — KPIs, sentiment/emotion distributions, reaction mix, comment timeline, and hourly heatmap
- **Deep Analysis** — Keyword clouds (overall, positive, negative), engagement-by-sentiment charts, and language diversity breakdown
- **Top Comments** — Most reacted, most engaged, most loved, and most laughed-at comments with Pareto analysis and near-duplicate clustering
- **Benchmarks** — Compares current post metrics against historical snapshots (percentile ranking across past analyses)
- **Content Intelligence** — Top hashtags, CTA detection, and readability scoring
- **Safety Panel** — Toxicity rate estimation, flagged high-engagement negative comments, and risk alerts
- **Post Planner** — Draft, schedule, and manage planned posts with saved reply templates
- **Story Sharing** — Generate shareable public links for analysis summaries

### Streamlit Dashboard (Lightweight)
- Token-based authentication (paste your Page Access Token)
- One-click comment fetching and analysis
- KPI cards, engagement highlights, and sentiment/emotion bar charts

---

## 🏗️ Project Structure

```
fb-comment-analyzer/
├── backend.py                 # Flask app — routes, OAuth, analysis engine, planner
├── dashboard.py               # Streamlit app — lightweight alternative UI
├── fetch_comments.py          # Facebook Graph API comment fetcher (v19.0)
├── model_utils.py             # HuggingFace model loaders (sentiment + emotion)
├── dashboard_components.py    # Streamlit chart/KPI components
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container build file
├── templates/
│   ├── base.html              # Base layout
│   ├── no_login.html          # Landing page (unauthenticated)
│   ├── choose_acc.html        # Account selection (user vs page)
│   ├── logged_in.html         # Post-login landing
│   ├── posts.html             # Post listing with thumbnails
│   ├── analysis_shell.html    # Analysis view container
│   └── views/
│       ├── overview.html      # Main analytics overview
│       ├── deep.html          # Deep analysis (keywords, charts)
│       ├── top_comments.html  # Top comments + clustering
│       ├── benchmark.html     # Historical benchmarks
│       ├── content_intel.html # Content intelligence panel
│       ├── safety.html        # Safety & toxicity panel
│       ├── planner.html       # Post planner & saved replies
│       ├── story.html         # Story generation
│       └── story_public.html  # Public story share page
└── static/
    ├── style.css              # Main stylesheet
    └── css/                   # Additional stylesheets
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- A **Facebook App** with the following permissions:
  - `public_profile`
  - `pages_read_engagement`
  - `pages_read_user_content`
- A **Page Access Token** (for page-level comment access)

### 1. Clone the Repository

```bash
git clone https://github.com/hamim00/fb-comment-analyzer.git
cd fb-comment-analyzer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download ~500MB of HuggingFace model weights for the sentiment and emotion classifiers.

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
FB_APP_ID=your_facebook_app_id
FB_APP_SECRET=your_facebook_app_secret
FB_REDIRECT_URI=http://localhost:5000/facebook/callback
SECRET_KEY=your_random_secret_key
ANA_SHARE_SECRET=your_share_secret
```

### 4. Run the Application

**Flask Web App** (full-featured):

```bash
python backend.py
```

Then open `http://localhost:5000` in your browser.

**Streamlit Dashboard** (lightweight):

```bash
streamlit run dashboard.py
```

Then open `http://localhost:8501` in your browser.

---

## 🔐 Authentication Flow (Flask App)

1. Click **Login with Facebook** on the landing page
2. Grant the requested permissions via Facebook OAuth
3. Choose between analyzing as your **User** account or a **Page** you manage
4. Browse your recent posts and click **Analyze** on any post
5. The app fetches all comments, runs NLP models, and renders the full analytics dashboard

---

## 🧠 Models Used

| Task | Model | Source |
|------|-------|--------|
| Sentiment Analysis | `cardiffnlp/twitter-roberta-base-sentiment-latest` | [HuggingFace](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) |
| Emotion Detection | `SamLowe/roberta-base-go_emotions` | [HuggingFace](https://huggingface.co/SamLowe/roberta-base-go_emotions) |

Both models run via the HuggingFace `transformers` pipeline with PyTorch backend and batch inference for efficiency.

---

## 📦 Dependencies

```
flask
streamlit
requests
pandas
numpy
scikit-learn
matplotlib
seaborn
transformers
torch
langdetect
python-dotenv
```

---

## 📸 Supported Post Types

The comment fetcher supports a wide range of Facebook URL formats:

- Standard posts (`/posts/`)
- Photos (`/photo.php?fbid=`)
- Videos (`/videos/`)
- Reels (`/reel/`)
- Watch videos (`/watch/?v=`)
- Group posts (`/groups/.../posts/`)
- Permalink and story URLs
- Raw Graph API object IDs

---

## 🧪 Demo Mode

For testing without a live Facebook token, the app includes built-in demo data:

```python
# In the Streamlit app, use:
# Token: test
# Post URL: test
```

This loads a small bilingual (English + Bengali) sample dataset so you can explore the dashboard without API access.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Mahmudul Hasan Hamim**

- GitHub: [@hamim00](https://github.com/hamim00)

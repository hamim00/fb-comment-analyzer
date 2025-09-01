# backend.py — polished Flask 3 app
from __future__ import annotations
import os, io, re, json, time, secrets, hashlib, sqlite3
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import (
    Flask, redirect, url_for, session, request,
    jsonify, render_template, Response
)

# -----------------------------------------------------------------------------
# ENV & APP
# -----------------------------------------------------------------------------
load_dotenv()
FB_APP_ID       = os.getenv("FB_APP_ID")
FB_APP_SECRET   = os.getenv("FB_APP_SECRET")
FB_REDIRECT_URI = os.getenv("FB_REDIRECT_URI")
SECRET_KEY      = os.getenv("SECRET_KEY", "dev-secret")
SHARE_SECRET    = os.getenv("ANA_SHARE_SECRET", "changeme-please")

app = Flask(__name__)
app.secret_key = SECRET_KEY

# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------
def parse_dt(v):
    """Best-effort to pandas Timestamp (UTC)."""
    if v is None:
        return pd.NaT
    try:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return pd.to_datetime(float(v), unit="s", utc=True, errors="coerce")
        return pd.to_datetime(v, utc=True, errors="coerce")
    except Exception:
        return pd.NaT

def _abbr(n):
    try: n = int(n or 0)
    except Exception: return "0"
    for thresh, suf in ((1_000_000_000,"B"),(1_000_000,"M"),(1_000,"K")):
        if n >= thresh:
            s = f"{n/thresh:.1f}".rstrip("0").rstrip(".")
            return f"{s}{suf}"
    return str(n)

app.jinja_env.filters["abbr"] = _abbr

# -----------------------------------------------------------------------------
# FACEBOOK: post list (choose_account -> show_posts)
# -----------------------------------------------------------------------------
def fetch_facebook_posts(token: str, page_id: str):
    url = f"https://graph.facebook.com/v19.0/{page_id}/posts"
    fields = (
        "id,permalink_url,message,created_time,full_picture,"
        "attachments{media_type,media,url,subattachments},"
        "reactions.type(LIKE).limit(0).summary(total_count),"
        "comments.limit(0).summary(total_count),"
        "shares"
    )
    res = requests.get(url, params={"access_token": token, "fields": fields, "limit": 10}).json()
    posts = []
    for p in res.get("data", []):
        picture_url, video_url = p.get("full_picture"), None
        for att in (p.get("attachments", {}) or {}).get("data", []) or []:
            mt, media = att.get("media_type"), att.get("media", {}) or {}
            if mt == "video":
                video_url = att.get("url")
                picture_url = media.get("image", {}).get("src") or picture_url
            elif mt == "photo":
                picture_url = media.get("image", {}).get("src") or picture_url
        like_count    = ((p.get("reactions") or {}).get("summary") or {}).get("total_count", 0)
        comment_count = ((p.get("comments")  or {}).get("summary") or {}).get("total_count", 0)
        share_count   = (p.get("shares") or {}).get("count", 0)
        posts.append({
            "id": p.get("id"),
            "permalink_url": p.get("permalink_url"),
            "message": p.get("message"),
            "created_time": p.get("created_time"),
            "picture_url": picture_url,
            "video_url": video_url,
            "like_count": like_count,
            "comment_count": comment_count,
            "share_count": share_count,
        })
    return posts

# -----------------------------------------------------------------------------
# AUTH FLOW
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    if not session.get("user"):
        return render_template("no_login.html")
    return redirect(url_for("show_posts"))

@app.route("/login")
def login():
    fb_auth_url = (
        f"https://www.facebook.com/v19.0/dialog/oauth"
        f"?client_id={FB_APP_ID}"
        f"&redirect_uri={FB_REDIRECT_URI}"
        f"&scope=public_profile,pages_read_engagement,pages_read_user_content"
        f"&response_type=code"
    )
    return redirect(fb_auth_url)

@app.route("/facebook/callback")
def facebook_callback():
    code = request.args.get("code")
    token_res = requests.get("https://graph.facebook.com/v19.0/oauth/access_token", params={
        "client_id": FB_APP_ID, "redirect_uri": FB_REDIRECT_URI, "client_secret": FB_APP_SECRET, "code": code
    }).json()
    access_token = token_res.get("access_token")
    if not access_token:
        return f"Failed to retrieve access token: {token_res}", 400

    user = requests.get("https://graph.facebook.com/v19.0/me",
                        params={"fields":"id,name,picture", "access_token": access_token}).json()
    session["user"] = user
    session["user_token"] = access_token

    pages = requests.get("https://graph.facebook.com/v19.0/me/accounts",
                         params={"access_token": access_token}).json().get("data", [])
    session["pages"] = pages
    return redirect(url_for("choose_account"))

@app.route("/choose_account", methods=["GET","POST"])
def choose_account():
    user  = session.get("user")
    pages = session.get("pages", [])
    if request.method == "POST":
        acc_type = request.form.get("account_type")
        page_id  = request.form.get("page_id")
        if acc_type == "user":
            session.update({"access_token": session["user_token"], "account_type":"user",
                            "account_name": user.get("name")})
            session.pop("page_id", None)
        elif acc_type == "page" and page_id:
            for p in pages:
                if p["id"] == page_id:
                    session.update({"access_token": p["access_token"], "account_type":"page",
                                    "account_name": p["name"], "page_id": p["id"]})
                    break
        return redirect(url_for("show_posts"))
    return render_template("choose_acc.html", user=user, pages=pages)

@app.route("/show_posts")
def show_posts():
    token        = session.get("access_token")
    account_type = session.get("account_type")
    user         = session.get("user")
    account_name = session.get("account_name")
    page_id = user["id"] if account_type == "user" else session.get("page_id")
    posts = fetch_facebook_posts(token, page_id) if (token and page_id) else []
    return render_template("posts.html", posts=posts, account_name=account_name,
                           account_type=account_type, access_token=token)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# -----------------------------------------------------------------------------
# COMMENT FETCH + ML
# -----------------------------------------------------------------------------
from fetch_comments import fetch_comments       # must include permalink_url if possible
from model_utils import load_sentiment_model, load_emotion_model

SENTIMENT_MODEL = load_sentiment_model()
EMOTION_MODEL   = load_emotion_model()

REACTION_COLS = ['like_count','love_count','haha_count','wow_count','sad_count','angry_count','care_count']
WORD_RE  = re.compile(r"[#@]?\w+", re.UNICODE)
URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.I)
EMOJI_RE = re.compile("["   # emoji-ish ranges
    "\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\u2600-\u26FF\u2700-\u27BF"
"]+", flags=re.UNICODE)

BN_STOP = set("""আমি আমরা তুমি আপনি তারা সে এই সেই ওই কোন কি বা আর এবং তবে কিন্তু তাই শুধু যেন যেই যে যদি যখন তবে যদিও যা যার যারাই যারা যেখানে যেখানে ছিল ছিলাম হবে করেছি করব করি করা করে করছেন করছেনা তো না নয় হবে না হয় হচ্ছে আছে ছিলেন হতে হতে পারে পর্যন্ত খুব অনেক আরো আরও আবার যেন তাহলে এখানে সেখানে তাদের সঙ্গে জন্য কারণ যার ফলে উপর নিচ আগে পরে মধ্যে আমার তোমার তার তাদের তোমাদের আমাদের আরেকটা কিছু কিছুটা কেউ কোনটা কোনগুলো তাহলে তবে তাই দিয়ে দিয়ে গেছে যায় হয়েই হল হলো হওয়া হওয়ার ছিলেন নই ছিলাম নাও নেন নেওয়া নেওয়া হয়েছে হয়নি কেন কিভাবে কেনো কেনও ইত্যাদি""".split())
EN_STOP = set("""a an and the of to in for on at by with about between into over from as is are was were be been being than then that this these those there here it its it's their theirs our ours your yours i me my mine we us you he him his she her they them who whom which what why how not no nor so such too very can will just do does did have has had up down out more most other some any each few own same than s t don should now""".split())

def ensure_engagement_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in REACTION_COLS:
        if c not in df.columns: df[c] = 0
    df['total_reactions'] = df[REACTION_COLS].sum(axis=1)
    if 'reply_count' not in df.columns: df['reply_count'] = 0
    df['engagement_score'] = df['total_reactions'] + df['reply_count']
    return df

def analyze_df(df: pd.DataFrame) -> pd.DataFrame:
    texts = df['comment_text'].fillna("").tolist()
    s_preds = SENTIMENT_MODEL(texts, batch_size=32, truncation=True, max_length=256)
    e_preds = EMOTION_MODEL(texts, batch_size=32, truncation=True, max_length=256, top_k=None)

    def top_label(p):
        if isinstance(p, list) and p:
            return sorted(p, key=lambda d: d.get('score', 0), reverse=True)[0].get('label', 'neutral')
        if isinstance(p, dict): return p.get('label','neutral')
        return 'neutral'

    df['sentiment'] = [(p[0]['label'] if isinstance(p, list) else p['label']).lower() for p in s_preds]
    df['emotion']   = [top_label(p).lower() for p in e_preds]
    df['emoji_count'] = df['comment_text'].astype(str).apply(lambda t: len(EMOJI_RE.findall(t)))
    df['word_count']  = df['comment_text'].astype(str).apply(lambda t: len(WORD_RE.findall(URL_RE.sub("", t))))
    return df

def tokenize(text: str):
    text = URL_RE.sub("", text.lower())
    toks = [t for t in WORD_RE.findall(text)]
    out = []
    for t in toks:
        if t.startswith('@') or t.startswith('#'):
            out.append(t); continue
        if t in EN_STOP or t in BN_STOP: continue
        if t.isdigit() or len(t) < 3: continue
        out.append(t)
    return out

# -----------------------------------------------------------------------------
# INSIGHTS
# -----------------------------------------------------------------------------
def compute_insights(df: pd.DataFrame):
    total_comments = int(len(df))
    unique_users   = int(df['user_id'].nunique() if 'user_id' in df else df['user_name'].nunique())
    total_react    = int(df['total_reactions'].sum())
    avg_react      = round(float(df['total_reactions'].mean() if total_comments else 0), 2)
    med_react      = float(df['total_reactions'].median() if total_comments else 0)
    med_eng        = float(df['engagement_score'].median() if total_comments else 0)
    emoji_ratio    = round(100 * (df['emoji_count'] > 0).mean(), 1) if 'emoji_count' in df else 0.0
    lang_series = (df['language'].fillna('unknown').astype(str).str.lower()
                   .replace({'bn':'bangla','en':'english'}))
    language_diversity = int(lang_series.nunique())

    sent_counts = df['sentiment'].value_counts().to_dict()
    pos = int(sent_counts.get('positive', 0))
    neg = int(sent_counts.get('negative', 0))
    neu = total_comments - pos - neg
    pos_ratio_among_polar = round(100 * pos / max(1, (pos + neg)), 1)
    top_emotion = (df['emotion'].value_counts().idxmax()
                   if 'emotion' in df and not df['emotion'].isna().all() else 'neutral')

    emo_counts = df['emotion'].value_counts().sort_values(ascending=False).head(7)
    emotions = {"labels": emo_counts.index.tolist(), "values": [int(x) for x in emo_counts.values.tolist()]}

    lang_counts = lang_series.value_counts().head(6)
    languages = {"labels": lang_counts.index.tolist(), "values": [int(x) for x in lang_counts.values.tolist()]}

    react_totals = {c.replace('_count','').upper(): int(df[c].sum()) for c in REACTION_COLS}
    react_mix = {"labels": list(react_totals.keys()), "values": list(react_totals.values())}

    if 'created_time' in df.columns:
        ts = pd.to_datetime(df['created_time'], errors='coerce', utc=True)
        by_day = ts.dt.tz_convert('UTC').dt.date.value_counts().sort_index()
        timeline = {"labels": [d.strftime("%Y-%m-%d") for d in pd.to_datetime(list(by_day.index))],
                    "values": [int(v) for v in by_day.values.tolist()]}
        best_hour = int(ts.dt.hour.value_counts().idxmax()) if ts.notna().any() else None

        tmin, tmax = ts.min(), ts.max()
        hours = max(1.0, (tmax - tmin).total_seconds() / 3600.0) if pd.notna(tmin) and pd.notna(tmax) else 1.0
        velocity = round(total_comments / hours, 2)

        hours_all = list(range(24))
        sent_hour = df.assign(hour=ts.dt.hour)
        def cnt(label):
            s = sent_hour[sent_hour['sentiment'].str.startswith(label, na=False)].groupby('hour')['sentiment'].count()
            return [int(s.get(h, 0)) for h in hours_all]
        hourly = {"labels": hours_all, "pos": cnt('pos'), "neg": cnt('neg'),
                  "neu": [int(total) - p - n for total, p, n in
                          zip(sent_hour.groupby('hour')['sentiment'].count().reindex(hours_all, fill_value=0),
                              cnt('pos'), cnt('neg'))]}
    else:
        timeline, best_hour, velocity, hourly = {"labels":[],"values":[]}, None, 0.0, {"labels":[], "pos":[], "neg":[], "neu":[]}

    def top_keywords(series: pd.Series, k=12):
        c = Counter()
        for t in series.astype(str): c.update(tokenize(t))
        return [{"word": w, "count": int(n)} for w,n in c.most_common(k)]
    keywords_all = top_keywords(df['comment_text'], k=15)
    keywords_pos = top_keywords(df[df['sentiment'].str.startswith('pos', na=False)]['comment_text'], k=10)
    keywords_neg = top_keywords(df[df['sentiment'].str.startswith('neg', na=False)]['comment_text'], k=10)
    top_keyword  = keywords_all[0]['word'] if keywords_all else ''

    def pick_rows(x, k=5):
        return [{"user": r['user_name'], "text": r['comment_text'],
                 "eng": int(r['engagement_score']), "react": int(r['total_reactions']),
                 "emotion": r['emotion'], "link": r.get('permalink_url', '')}
                for _,r in x.head(k).iterrows()]
    top_haha     = pick_rows(df.sort_values('haha_count', ascending=False).query("haha_count > 0"), 5)
    top_reacted  = pick_rows(df.sort_values('total_reactions', ascending=False), 5)
    top_engaged  = pick_rows(df.sort_values('engagement_score', ascending=False), 5)
    top_pos      = pick_rows(df[df['sentiment'].str.startswith('pos', na=False)].sort_values('engagement_score', ascending=False), 3)
    top_neg      = pick_rows(df[df['sentiment'].str.startswith('neg', na=False)].sort_values('engagement_score', ascending=False), 3)
    examples     = {"positive": top_pos, "negative": top_neg}

    risk = (df[df['sentiment'].str.startswith('neg', na=False)]
            .query("engagement_score >= @df.engagement_score.quantile(0.75)")
            .sort_values('engagement_score', ascending=False).head(5))
    risks = pick_rows(risk, 5)

    safety_score = max(0, 100 - round(100 * (neg / max(1, total_comments)), 1))
    advocacy     = round(100 * (pos / max(1, total_comments)), 1)

    return {
        "kpis": {
            "total_comments": total_comments,
            "unique_users": unique_users,
            "total_reactions": total_react,
            "avg_reactions_per_comment": avg_react,
            "median_reactions": med_react,
            "median_engagement": med_eng,
            "emoji_pct": emoji_ratio,
            "language_diversity": language_diversity,
            "positivity_ratio": pos_ratio_among_polar,
            "safety_score": safety_score,
            "advocacy": advocacy,
            "best_hour": best_hour,
            "velocity_cph": velocity,
            "top_emotion": top_emotion,
            "top_keyword": top_keyword
        },
        "sentiment": {"labels": ["Positive","Negative","Neutral"], "values": [pos, neg, max(0, neu)]},
        "emotions": emotions,
        "languages": languages,
        "reactions": react_mix,
        "timeline": timeline,
        "hourly": hourly,
        "top_commenters_count": [],
        "top_commenters_eng":  [],
        "keywords_all": keywords_all,
        "keywords_pos": keywords_pos,
        "keywords_neg": keywords_neg,
        "examples": examples,
        "top_haha": top_haha,
        "top_reacted": top_reacted,
        "top_engaged": top_engaged,
        "risks": risks
    }

# -----------------------------------------------------------------------------
# History & baselines (for deltas)
# -----------------------------------------------------------------------------
HIST_PATH = os.path.join(os.path.dirname(__file__), "data_insights_history.json")

def _history_load():
    if os.path.exists(HIST_PATH):
        try: return json.load(open(HIST_PATH,"r",encoding="utf-8"))
        except: return {}
    return {}

def _history_save(hist: dict):
    with open(HIST_PATH,"w",encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)

def compute_benchmarks(post_id: str, current: dict):
    hist = _history_load()
    hist[post_id] = {"ts": int(time.time()), "kpis": current.get("kpis", {})}
    _history_save(hist)
    items = [v["kpis"] for _,v in list(hist.items())[-20:] if isinstance(v, dict) and v.get("kpis")]
    def pct_of(metric, value):
        vals = sorted([x.get(metric, 0) for x in items if metric in x])
        if not vals: return None
        below = sum(1 for v in vals if v <= value)
        return round(100 * below / len(vals), 1)
    k = current.get("kpis", {})
    percentiles = {m: pct_of(m, k.get(m,0)) for m in ("advocacy","safety_score","avg_reactions_per_comment","total_comments")}
    return {"percentiles": percentiles, "count_history": len(items)}

def compute_quality(current: dict):
    now = int(time.time())
    return {"data_freshness": "fresh", "coverage_pct": 100,
            "model_version": "sent: cardiffnlp/twitter-roberta · emo: go_emotions",
            "generated_at": now}

def build_highlights(data: dict):
    k = data.get("kpis", {}); out=[]
    if "advocacy" in k: out.append(f"🎯 Advocacy {k['advocacy']}%")
    if "safety_score" in k: out.append(f"🛡️ Safety {k['safety_score']}")
    if "best_hour" in k and k["best_hour"] is not None: out.append(f"⏰ Best hour {k['best_hour']:02d}:00 UTC")
    if "velocity_cph" in k: out.append(f"⚡ {k['velocity_cph']} comments/hour")
    if "top_emotion" in k: out.append(f"💬 Top emotion {k['top_emotion']}")
    if "top_keyword" in k and k["top_keyword"]: out.append(f"#️⃣ {k['top_keyword']}")
    return out

# -----------------------------------------------------------------------------
# Extra analytics for Top Comments (KPIs + 4 lists)
# -----------------------------------------------------------------------------
def compute_top_comments_panel(df: pd.DataFrame, insights: dict):
    if df is None or df.empty:
        return {"kpis": {}, "lists": {"top_reacted":[],"top_engaged":[],"top_loved":[],"top_haha":[]}}

    # base cols
    for c in REACTION_COLS:
        if c not in df.columns: df[c]=0
    if 'total_reactions' not in df.columns:
        df['total_reactions'] = df[REACTION_COLS].sum(axis=1)
    if 'reply_count' not in df.columns:
        df['reply_count'] = 0
    if 'engagement_score' not in df.columns:
        df['engagement_score'] = df['total_reactions'] + df['reply_count']

    k = insights.get("kpis", {}) if insights else {}

    total = int(len(df))
    avg_reacts = float(df['total_reactions'].mean() if total else 0)
    pos = int((df['sentiment'].astype(str).str.startswith('pos')).sum())
    neg = int((df['sentiment'].astype(str).str.startswith('neg')).sum())
    polar = pos + neg
    positivity = round(100 * pos / max(1, polar), 1)
    questions = int(df['comment_text'].astype(str).str.contains(r'\?', regex=True).sum())
    high_thresh = df['engagement_score'].quantile(0.75) if total else 0
    high_eng = int((df['engagement_score'] >= high_thresh).sum())
    unanswered = int((df.get('reply_count', 0) == 0).sum())

    def _pick_rows(df_slice, cap=5):
        return [{
            "user": r.get('user_name',''),
            "text": r.get('comment_text',''),
            "sentiment": r.get('sentiment',''),
            "emotion": r.get('emotion',''),
            "react": int(r.get('total_reactions', 0)),
            "eng": int(r.get('engagement_score', 0)),
            "ts": str(r.get('created_time','')) if 'created_time' in r else '',
            "link": r.get('permalink_url','')
        } for _, r in df_slice.head(cap).iterrows()]

    lists = {
        "top_reacted": _pick_rows(df.sort_values('total_reactions', ascending=False), 5),
        "top_engaged": _pick_rows(df.sort_values('engagement_score', ascending=False), 5),
        "top_loved":   _pick_rows(df.sort_values('love_count', ascending=False).query("love_count > 0"), 5),
        "top_haha":    _pick_rows(df.sort_values('haha_count', ascending=False).query("haha_count > 0"), 5),
    }

    return {
        "kpis": dict(
            total=total,
            avg_reacts=round(avg_reacts,2),
            positivity=positivity,
            negatives=neg,
            questions=questions,
            high_eng=int(high_eng),
            unanswered=unanswered
        ),
        "lists": lists
    }

# -----------------------------------------------------------------------------
# In-process cache per post
# -----------------------------------------------------------------------------
_POST_CACHE = {}

def _get_post_bundle(token: str, post_id: str):
    if post_id in _POST_CACHE:
        return _POST_CACHE[post_id]

    df = fetch_comments(token, post_id)
    if df is None or getattr(df, "empty", True):
        bundle = (pd.DataFrame(), {}, [])
        _POST_CACHE[post_id] = bundle
        return bundle

    df = ensure_engagement_columns(df.copy())
    df = analyze_df(df)
    if 'created_time' in df.columns:
        df['created_time'] = df['created_time'].apply(parse_dt)

    data = compute_insights(df)

    cols = [
        'user_name','comment_text','sentiment','emotion',
        'total_reactions','engagement_score','reply_count',
        'created_time','language','permalink_url'
    ]
    have = [c for c in cols if c in df.columns]
    rows = df[have].to_dict(orient="records")

    bundle = (df, data, rows)
    _POST_CACHE[post_id] = bundle
    return bundle

# -----------------------------------------------------------------------------
# Deep/time analytics
# -----------------------------------------------------------------------------
def _parse_iso(dt_str):
    if not dt_str: return None
    try: return pd.to_datetime(dt_str, utc=True)
    except Exception: return None

def _apply_time_window(df: pd.DataFrame, start: str|None, end: str|None, rng: str|None):
    if 'created_time' not in df.columns: return df
    ts = pd.to_datetime(df['created_time'], utc=True, errors='coerce')
    df = df.assign(_ts=ts)
    tmin, tmax = ts.min(), ts.max()

    s = _parse_iso(start); e = _parse_iso(end)
    if rng and not (s or e):
        try:
            q = rng.strip().lower()
            amt = int(''.join(c for c in q if c.isdigit()))
            delta = pd.Timedelta(days=amt) if 'd' in q else pd.Timedelta(hours=amt)
            s = (tmax - delta) if pd.notna(tmax) else None
            e = tmax
        except Exception:
            pass
    if not s: s = tmin
    if not e: e = tmax
    if s is not None and e is not None:
        df = df[(df['_ts'] >= s) & (df['_ts'] <= e)]
    return df.drop(columns=['_ts'])

def _time_bin(span_hours: float):
    return '15min' if span_hours <= 6 else 'H'

def _time_analytics(df: pd.DataFrame):
    if df.empty or 'created_time' not in df.columns:
        return {"timeline":{"labels":[],"values":[]},
                "emotion_stack":{"labels":[],"series":[]},
                "spikes":[],
                "half_life_hours":None,"t_first10_hours":None,
                "contagion":{"labels":[],"matrix":[]},
                "reaction_curve":{"labels":[],"values":[]}}
    dt = pd.to_datetime(df['created_time'], utc=True, errors='coerce')
    df = df.assign(_ts=dt)
    span_hours = max(1e-6, (dt.max() - dt.min()).total_seconds()/3600.0)
    freq = _time_bin(span_hours)

    g = (df.set_index('_ts').groupby(pd.Grouper(freq=freq)).size().rename('count'))
    tl_labels = [x.strftime('%Y-%m-%d %H:%M') for x in g.index]
    tl_values = [int(v) for v in g.values]

    emo_total = df['emotion'].value_counts().head(5).index.tolist()
    emo = (df.set_index('_ts').groupby([pd.Grouper(freq=freq), 'emotion']).size()
             .unstack(fill_value=0))
    emo = emo[[c for c in emo.columns if c in emo_total]]
    emo_series = [{"name": c, "values": [int(x) for x in emo[c].values.tolist()]} for c in emo.columns]
    emo_labels = [x.strftime('%H:%M' if freq=='15min' else '%Y-%m-%d %H:%M') for x in emo.index]

    spikes = []
    for c in emo.columns:
        s = emo[c].astype(float)
        mu = s.rolling(3, min_periods=3).mean()
        sd = s.rolling(3, min_periods=3).std()
        z = (s - mu) / (sd.replace(0, np.nan))
        for idx, val in z.dropna().items():
            if val >= 2.0 and s.loc[idx] >= 2:
                spikes.append({"t": idx.strftime('%Y-%m-%d %H:%M'),
                               "emotion": c, "z": float(round(val,2)),
                               "value": int(s.loc[idx])})
    spikes = sorted(spikes, key=lambda x: x['z'], reverse=True)[:8]

    y = np.array(tl_values, dtype=float)
    if len(y) >= 4:
        peak_i = int(np.argmax(y))
        x_tail = np.arange(len(y) - peak_i, dtype=float)
        y_tail = y[peak_i:].clip(min=1e-6)
        b, a = np.polyfit(x_tail, np.log(y_tail), 1)
        y_fit = np.exp(a + b * x_tail)
        tau = float(round(-1.0 / b, 2)) if b < 0 else None
        fit_full = [None]*peak_i + [float(round(v,2)) for v in y_fit.tolist()]
    else:
        fit_full, tau = [None]*len(y), None

    cum = np.cumsum(y)
    half = None
    if cum[-1] > 0:
        thresh = 0.5*cum[-1]; idx = np.argmax(cum >= thresh)
        half = float(round(idx * (0.25 if _time_bin(span_hours)=='15min' else 1.0), 2))
    t10 = None
    if len(cum) and cum[-1] >= 10:
        idx10 = np.argmax(cum >= 10)
        t10 = float(round(idx10 * (0.25 if _time_bin(span_hours)=='15min' else 1.0), 2))

    labmap = df['sentiment'].fillna('neutral').str.lower().map(
        lambda s: 'positive' if s.startswith('pos') else ('negative' if s.startswith('neg') else 'neutral')
    )
    d2 = df.assign(_sent=labmap).sort_values('_ts')
    base_counts = d2['_sent'].value_counts()
    base_prob = (base_counts / base_counts.sum()).to_dict()
    labels = ['positive','neutral','negative']
    trans = {(i,j):0 for i in labels for j in labels}
    prev_t, prev_s = None, None
    for _, r in d2.iterrows():
        t, s = r['_ts'], r['_sent']
        if prev_t is not None and (t - prev_t).total_seconds() <= 1800:
            trans[(prev_s, s)] += 1
        prev_t, prev_s = t, s
    matrix = []
    for i in labels:
        row_total = sum(trans[(i,j)] for j in labels) or 1
        row = []
        for j in labels:
            p_next = trans[(i,j)]/row_total
            base = base_prob.get(j, 1e-9)
            row.append(round(p_next/base, 2))
        matrix.append(row)

    base_t = dt.min()
    age_h = ((dt - base_t).dt.total_seconds() // 3600).astype(int)
    df2 = df.assign(_age=age_h)
    for c in REACTION_COLS:
        if c not in df2.columns: df2[c]=0
    df2['total_reactions'] = df2[REACTION_COLS].sum(axis=1)
    rc = df2.groupby('_age')['total_reactions'].mean().sort_index()
    rc_labels = [int(i) for i in rc.index.tolist()]
    rc_values = [float(round(v,2)) for v in rc.values.tolist()]

    return {"timeline":{"labels":tl_labels,"values":tl_values,"fit":fit_full,"tau_bins":tau},
            "emotion_stack":{"labels":emo_labels,"series":emo_series},
            "spikes":spikes,
            "half_life_hours":half,"t_first10_hours":t10,
            "contagion":{"labels":labels,"matrix":matrix},
            "reaction_curve":{"labels":rc_labels,"values":rc_values}}

# -----------------------------------------------------------------------------
# VIEWS
# -----------------------------------------------------------------------------
VALID_VIEWS = {
    "overview": "Overview",
    "deep": "Deep Analysis",
    "top_comments": "Top Comments",
    "content_intel": "Content Intelligence",
    "safety": "Safety & Moderation",
    "benchmark": "Benchmark & Percentiles",
    "quality": "Quality & Provenance",
    "story": "Story Builder",
    "planner": "Content Planner",
}

@app.route("/analyze/<post_id>")
def analyze_page(post_id):
    return redirect(url_for("analyze_view", post_id=post_id, view="overview"))

@app.route("/analyze/<post_id>/<view>")
def analyze_view(post_id, view):
    if view not in VALID_VIEWS:
        return redirect(url_for("analyze_view", post_id=post_id, view="overview"))
    token = session.get("access_token")
    if not token:
        return redirect(url_for("login"))

    df, data, rows = _get_post_bundle(token, post_id)

    extras = {}
    if view == "content_intel":
        extras = compute_content_intel(df) if df is not None and not df.empty else {}
    elif view == "safety":
        extras = compute_safety_panel(df, data) if df is not None and not df.empty else {}
    elif view == "benchmark":
        extras = compute_benchmarks(post_id, data)
    elif view == "quality":
        extras = compute_quality(data)
    elif view == "story":
        extras = {"highlights": build_highlights(data)}
    elif view == "planner":
        extras = {"drafts": planner_list()}
    elif view == "top_comments":
        extras = compute_top_comments_panel(df, data)

    try:
        server_ts = _time_analytics(df) if df is not None and not df.empty else {}
    except Exception:
        server_ts = {}

    return render_template("analysis_shell.html",
        view=view, view_title=VALID_VIEWS[view],
        post_id=post_id, data=data, rows=rows, extras=extras,
        server_ts=server_ts, has_error=(df is None or df.empty),
    )

# -----------------------------------------------------------------------------
# JSON APIs
# -----------------------------------------------------------------------------
@app.get("/api/analyze")
def api_analyze():
    token = session.get('access_token')
    post_id = request.args.get("post_id")
    if not token or not post_id:
        return jsonify({"error":"missing token/post_id"}), 400
    df = fetch_comments(token, post_id)
    if df is None or getattr(df,"empty",True):
        return jsonify({"data": {}, "rows": []})
    df = ensure_engagement_columns(df.copy())
    df = analyze_df(df)
    if 'created_time' in df.columns:
        df['created_time'] = df['created_time'].apply(parse_dt)
    data = compute_insights(df)
    cols = ['user_name','comment_text','sentiment','emotion','total_reactions',
            'engagement_score','reply_count','created_time','language','permalink_url']
    have = [c for c in cols if c in df.columns]
    rows = df[have].to_dict(orient="records")
    return jsonify({"data": data, "rows": rows})

@app.get("/api/analyze_window")
def api_analyze_window():
    token = session.get('access_token')
    post_id = request.args.get("post_id")
    rng  = request.args.get("range")
    start= request.args.get("start")
    end  = request.args.get("end")
    if not token or not post_id:
        return jsonify({"error":"missing token/post_id"}), 400
    df, _, _ = _get_post_bundle(token, post_id)
    if df is None or df.empty:
        return jsonify({"data": {}, "rows": [], "ts": {}})

    dfw = _apply_time_window(df.copy(), start, end, rng)
    if dfw.empty:
        return jsonify({"data": {}, "rows": [], "ts": {}})

    d_ins = compute_insights(dfw)
    ts_ins = _time_analytics(dfw)
    cols = ['user_name','comment_text','sentiment','emotion','total_reactions','engagement_score','permalink_url']
    have = [c for c in cols if c in dfw.columns]
    rows_w = dfw[have].to_dict(orient="records")
    return jsonify({"data": d_ins, "rows": rows_w, "ts": ts_ins})

@app.get("/export/comments/<post_id>.csv")
def export_comments_csv(post_id):
    token = session.get('access_token')
    if not token: return Response("login required", status=401)
    df = fetch_comments(token, post_id)
    if df is None or getattr(df, "empty", True):
        return Response("no data", status=404)
    df = ensure_engagement_columns(df.copy())
    df = analyze_df(df)
    buf = io.StringIO(); df.to_csv(buf, index=False); buf.seek(0)
    return Response(
        buf.getvalue().encode('utf-8-sig'),
        headers={"Content-Type":"text/csv; charset=utf-8",
                 "Content-Disposition": f"attachment; filename=comments_{post_id}.csv"}
    )

# -----------------------------------------------------------------------------
# Content Intel & Safety panels
# -----------------------------------------------------------------------------
def compute_content_intel(df: pd.DataFrame):
    def hashtags(text): return [w for w in str(text).split() if w.startswith("#")]
    tag_counts = {}
    for t in df['comment_text'].fillna(""):
        for h in hashtags(t): tag_counts[h] = tag_counts.get(h, 0) + 1
    top_tags = sorted([{"tag":k,"count":v} for k,v in tag_counts.items()],
                      key=lambda x: -x["count"])[:10]
    cta_words = ("buy","order","subscribe","follow","share","signup","call","dm")
    cta_hits  = int(df['comment_text'].str.lower().str.contains("|".join(cta_words), regex=True, na=False).sum())
    word_lens = df['comment_text'].fillna("").str.split().apply(lambda ws: sum(len(w) for w in ws)/max(1,len(ws)))
    readability = float(round(word_lens.mean(), 2))
    return {"top_tags": top_tags, "cta_hits": cta_hits, "readability": readability}

def compute_safety_panel(df: pd.DataFrame, insights: dict):
    bad = ("stupid","idiot","hate","kill","trash","shame","fraud","scam","liar","fake")
    tox = df['comment_text'].str.lower().str.contains("|".join(bad), regex=True, na=False)
    tox_rate = round(100 * tox.mean(), 1)
    recent = df.sort_values('created_time', ascending=False).head(200)
    tox_recent = recent[recent['comment_text'].str.lower().str.contains("|".join(bad), regex=True, na=False)]
    examples = tox_recent.sort_values("engagement_score", ascending=False).head(5)
    examples = [{"user": r["user_name"], "text": r["comment_text"], "eng": int(r["engagement_score"]),
                 "link": r.get("permalink_url","")} for _,r in examples.iterrows()]
    return {"tox_rate": tox_rate, "tox_examples": examples, "risks": insights.get("risks", [])}

# -----------------------------------------------------------------------------
# Story share (assist mode)
# -----------------------------------------------------------------------------
def _sign_token(payload: dict) -> str:
    body = json.dumps(payload, sort_keys=True)
    sig = hashlib.sha256((body + SHARE_SECRET).encode()).hexdigest()[:16]
    return secrets.token_urlsafe(10) + "." + sig + "." + str(int(time.time()))

@app.post("/story/generate/<post_id>")
def story_generate(post_id):
    token = _sign_token({"post_id": post_id})
    return jsonify({"url": url_for("story_share", token=token, _external=True)})

@app.get("/story/s/<token>")
def story_share(token):
    post_id = request.args.get("post_id")
    return render_template("views/story_public.html", token=token, post_id=post_id)

# -----------------------------------------------------------------------------
# Planner + Saved Replies + Quick Schedule
# -----------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "planner.db")

def _db():
    con = sqlite3.connect(DB_PATH); con.row_factory = sqlite3.Row
    return con

def planner_init():
    con = _db()
    con.execute("""
    CREATE TABLE IF NOT EXISTS planned_posts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      platform TEXT,
      target   TEXT,
      caption  TEXT,
      hashtags TEXT,
      media_url TEXT,
      scheduled_at TEXT,
      status TEXT DEFAULT 'draft',
      created_at TEXT DEFAULT (DATETIME('now'))
    )""")
    con.execute("""
    CREATE TABLE IF NOT EXISTS saved_replies (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      text TEXT NOT NULL,
      tag  TEXT,
      created_at TEXT DEFAULT (DATETIME('now'))
    )""")
    con.commit(); con.close()

try: planner_init()
except Exception as e: print("planner_init error:", e)

def planner_list():
    con = _db()
    rows = [dict(r) for r in con.execute("SELECT * FROM planned_posts ORDER BY id DESC").fetchall()]
    con.close(); return rows

@app.post("/planner/new")
def planner_new():
    d = request.form
    con = _db()
    con.execute("""INSERT INTO planned_posts(platform,target,caption,hashtags,media_url,scheduled_at,status)
                   VALUES(?,?,?,?,?,?,?)""",
                (d.get("platform"), d.get("target"), d.get("caption"), d.get("hashtags"),
                 d.get("media_url"), d.get("scheduled_at"), "draft"))
    con.commit(); con.close()
    return redirect(request.referrer or url_for("analyze_view",
                    post_id=request.args.get("post_id"), view="planner"))

@app.post("/planner/delete/<int:pid>")
def planner_delete(pid):
    con = _db(); con.execute("DELETE FROM planned_posts WHERE id=?", (pid,))
    con.commit(); con.close()
    return redirect(request.referrer or "/")

@app.get("/planner/open_composer/<int:pid>")
def planner_open(pid):
    con = _db(); row = con.execute("SELECT * FROM planned_posts WHERE id=?", (pid,)).fetchone()
    con.close()
    if not row: return redirect(request.referrer or "/")
    return redirect("https://www.facebook.com/")

# ---- Saved replies API ----
@app.get("/api/saved_replies")
def api_saved_replies_list():
    con = _db()
    rows = [dict(r) for r in con.execute("SELECT * FROM saved_replies ORDER BY id DESC").fetchall()]
    con.close(); return jsonify({"items": rows})

@app.post("/api/saved_replies")
def api_saved_replies_add():
    d = request.get_json(silent=True) or request.form
    text = (d.get("text") or "").strip()
    tag  = (d.get("tag") or "").strip()
    if not text: return jsonify({"error":"text required"}), 400
    con = _db(); cur = con.execute("INSERT INTO saved_replies(text, tag) VALUES(?,?)", (text, tag))
    con.commit(); con.close()
    return jsonify({"ok": True, "id": cur.lastrowid})

@app.delete("/api/saved_replies/<int:rid>")
def api_saved_replies_delete(rid):
    con = _db(); con.execute("DELETE FROM saved_replies WHERE id=?", (rid,))
    con.commit(); con.close()
    return jsonify({"ok": True})

# ---- Schedule reply inline (no navigation) ----
@app.post("/api/schedule_reply")
def api_schedule_reply():
    d = request.get_json(silent=True) or request.form
    post_id = d.get("post_id")
    reply   = (d.get("reply") or "").strip()
    when    = d.get("when")  # ISO string or None
    if not post_id or not reply:
        return jsonify({"error":"post_id and reply required"}), 400
    con = _db()
    con.execute("""INSERT INTO planned_posts(platform,target,caption,hashtags,media_url,scheduled_at,status)
                   VALUES(?,?,?,?,?,?,?)""",
                ("facebook", f"reply:{post_id}", reply, "", "", when or "", "scheduled"))
    con.commit(); con.close()
    return jsonify({"ok": True})

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)

# fetch_comments.py
import re
import time
from io import StringIO
from typing import Optional, Dict

import pandas as pd
import requests

# Optional language detection
try:
    from langdetect import detect, LangDetectException  # type: ignore
except Exception:  # langdetect not installed or unavailable
    detect = None

    class LangDetectException(Exception):
        ...


# ---------- Reliable GET with simple backoff ----------
def _get(url: str, params: dict | None = None, timeout: int = 30, tries: int = 3) -> requests.Response:
    """
    Wrapper around requests.get with basic retries for transient errors.
    Retries on 429 and 5xx responses with incremental backoff (2s, 4s, 6s).
    Returns the first non-transient response; if all fail, returns the last response.
    """
    for i in range(tries):
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 * (i + 1))
            continue
        return r
    return r  # likely an error; caller should inspect


# ---------- URL → Object ID extractor (reels/videos/photos/posts/groups) ----------
_PATTERNS = [
    (re.compile(r"facebook\.com/reel/(\d+)", re.I), lambda m: m.group(1)),
    (re.compile(r"facebook\.com/.*/videos/(\d+)", re.I), lambda m: m.group(1)),
    (re.compile(r"facebook\.com/watch/\?v=(\d+)", re.I), lambda m: m.group(1)),
    (re.compile(r"[?&]v=(\d+)", re.I), lambda m: m.group(1)),
    (re.compile(r"facebook\.com/photo\.php\?[^#]*\bfbid=(\d+)", re.I), lambda m: m.group(1)),
    (re.compile(r"facebook\.com/(?:permalink|story)\.php\?[^#]*\bstory_fbid=(\d+)", re.I), lambda m: m.group(1)),
    (re.compile(r"facebook\.com/.*/posts/(\d+)", re.I), lambda m: m.group(1)),
    # groups → Graph expects "<groupId>_<postId>"
    (re.compile(r"facebook\.com/groups/(\d+)/permalink/(\d+)", re.I), lambda m: f"{m.group(1)}_{m.group(2)}"),
    (re.compile(r"facebook\.com/groups/(\d+)/posts/(\d+)", re.I), lambda m: f"{m.group(1)}_{m.group(2)}"),
]


def extract_object_id(url: str) -> Optional[str]:
    """Extract a Graph object ID from a variety of Facebook URL shapes."""
    if not url:
        return None
    url = url.strip()
    for rx, fmt in _PATTERNS:
        m = rx.search(url)
        if m:
            return fmt(m)
    # allow raw numeric IDs as input
    if re.fullmatch(r"\d+", url):
        return url
    return None


# ---------- Demo/Test data ----------
def load_test_comments() -> pd.DataFrame:
    data = StringIO(
        """
comment_id,user_id,user_name,comment_text,created_time,like_count,love_count,haha_count,wow_count,sad_count,angry_count,care_count,reply_count,user_profile_link,user_gender,is_verified,language,parent_comment_id
cmt_1,user_1,নুসরাত জাহান শিলা,"খুব সুন্দর লিখেছেন, মুগ্ধ হলাম। শুভকামনা।",2024-06-09 12:01:15,8,3,0,1,0,0,0,1,https://facebook.com/101001,Female,False,Bangla,
cmt_2,user_2,John Abraham,"This is such an informative post. Thanks for sharing!",2024-06-09 12:05:50,5,2,0,0,0,0,1,0,https://facebook.com/101002,Male,False,English,
cmt_3,user_3,সাইফুল ইসলাম মেহেদী,"ভাই, এসব ফালতু কথা বাদ দেন। সময় নষ্ট!",2024-06-09 12:12:30,1,0,2,0,0,2,0,2,https://facebook.com/101003,Male,False,Bangla,
"""
    )
    # (Trimmed for brevity; your original long CSV block is fine to keep)
    return pd.read_csv(data)


# ---------- Facebook fetcher ----------
def fetch_comments_from_facebook(token: str, post_url: str) -> pd.DataFrame:
    """
    Fetches top-level comments for a given Facebook object URL/ID.
    Returns a DataFrame with:
      comment_id, user_id, user_name, comment_text, created_time,
      like_count, love_count, haha_count, wow_count, sad_count, angry_count, care_count,
      reply_count, user_profile_link, user_gender, is_verified, language, parent_comment_id
    """
    oid = extract_object_id(post_url)
    if not oid:
        print(f"[ERROR] Could not extract an object ID from: {post_url}")
        return pd.DataFrame()

    # Helpers kept inside to avoid global namespace noise
    def get_reaction_counts(comment_id: str, token: str) -> Dict[str, int]:
        types = ["LIKE", "LOVE", "HAHA", "WOW", "SAD", "ANGRY", "CARE"]
        counts: Dict[str, int] = {}
        for reaction in types:
            try:
                r = _get(
                    f"https://graph.facebook.com/v19.0/{comment_id}/reactions",
                    params={"type": reaction, "summary": "total_count", "limit": 0, "access_token": token},
                    timeout=20,
                )
                j = r.json()
                counts[f"{reaction.lower()}_count"] = j.get("summary", {}).get("total_count", 0)
            except Exception as ex:
                print(f"[WARN] Could not fetch reaction '{reaction}' for {comment_id}: {ex}")
                counts[f"{reaction.lower()}_count"] = 0
        return counts

    def get_user_info(user_id: str, token: str) -> Dict[str, str]:
        try:
            r = _get(
                f"https://graph.facebook.com/v19.0/{user_id}",
                params={"fields": "link,verified", "access_token": token},
                timeout=20,
            )
            j = r.json()
            return {"user_profile_link": j.get("link", ""), "is_verified": j.get("verified", "")}
        except Exception as ex:
            print(f"[WARN] Could not fetch user info for {user_id}: {ex}")
            return {"user_profile_link": "", "is_verified": ""}

    url = f"https://graph.facebook.com/v19.0/{oid}/comments"
    params = {
        "access_token": token,
        "fields": "id,message,from,created_time,comment_count,parent",
        "filter": "toplevel",  # change to 'stream' if you want replies too
        "limit": 100,
        "summary": "true",
    }

    all_rows: list[dict] = []
    page = 1

    while url:
        print(f"[INFO] Requesting page {page} of comments: {url}")
        try:
            r = _get(url, params=params if page == 1 else None, timeout=30)
            payload = r.json()
        except Exception as ex:
            print(f"[ERROR] Exception during API call: {ex}")
            break

        if r.status_code != 200 or "error" in payload:
            err = payload.get("error", {})
            print(
                f"[ERROR] Graph API {r.status_code}: {err.get('type')} {err.get('code')} – {err.get('message')}"
            )
            break

        data = payload.get("data", [])
        print(f"[INFO] Fetched {len(data)} comments on this page.")

        for c in data:
            cid = c.get("id", "")
            user = c.get("from") or {}
            message = c.get("message") or ""
            uid = user.get("id", "")
            uname = user.get("name", "")

            rx = get_reaction_counts(cid, token)
            uex = get_user_info(uid, token) if uid else {"user_profile_link": "", "is_verified": ""}

            # language detection (optional)
            if detect:
                try:
                    lang = detect(message) if message else ""
                except LangDetectException:
                    lang = ""
            else:
                lang = ""

            all_rows.append(
                {
                    "comment_id": cid,
                    "user_id": uid,
                    "user_name": uname,
                    "comment_text": message,
                    "created_time": c.get("created_time", ""),
                    "like_count": rx.get("like_count", 0),
                    "love_count": rx.get("love_count", 0),
                    "haha_count": rx.get("haha_count", 0),
                    "wow_count": rx.get("wow_count", 0),
                    "sad_count": rx.get("sad_count", 0),
                    "angry_count": rx.get("angry_count", 0),
                    "care_count": rx.get("care_count", 0),
                    "reply_count": c.get("comment_count", 0),
                    "user_profile_link": uex.get("user_profile_link", ""),
                    "user_gender": "",  # keep column for compatibility; Graph rarely returns gender
                    "is_verified": uex.get("is_verified", ""),
                    "language": lang,
                    "parent_comment_id": (c.get("parent") or {}).get("id", ""),
                }
            )

        url = (payload.get("paging") or {}).get("next")
        params = None  # only send params for the first page
        page += 1

    if not all_rows:
        print("[WARN] No comments found. Check token, permissions, or post visibility.")
        return pd.DataFrame()

    df = pd.DataFrame(
        all_rows,
        columns=[
            "comment_id",
            "user_id",
            "user_name",
            "comment_text",
            "created_time",
            "like_count",
            "love_count",
            "haha_count",
            "wow_count",
            "sad_count",
            "angry_count",
            "care_count",
            "reply_count",
            "user_profile_link",
            "user_gender",
            "is_verified",
            "language",
            "parent_comment_id",
        ],
    )
    print(f"[SUCCESS] Total comments fetched: {len(df)}")
    return df


# ---------- Public wrapper used by dashboard ----------
def fetch_comments(token: str, post_url: str) -> pd.DataFrame:
    """
    Wrapper used by the app:
    - If token/post_url are both "test", returns built-in demo comments.
    - Otherwise calls the live Facebook fetcher.
    """
    if token == "test" and post_url == "test":
        return load_test_comments()
    return fetch_comments_from_facebook(token, post_url)

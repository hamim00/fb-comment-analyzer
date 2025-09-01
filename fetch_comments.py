# fetch_comments.py (updated, investor-page ready)
# NOTE: drop-in replacement. Keeps your public function names intact.
from __future__ import annotations

import re
import time
from io import StringIO
from typing import Optional, Dict, List, Any

import pandas as pd
import requests

GRAPH = "https://graph.facebook.com/v19.0"

# ---------- Optional language detection ----------
try:
    from langdetect import detect, LangDetectException  # type: ignore
except Exception:  # langdetect not installed or unavailable
    detect = None

    class LangDetectException(Exception):
        pass


# ---------- Small GET with retries ----------
def _get(url: str, params: dict | None = None, timeout: int = 30, tries: int = 3) -> requests.Response:
    """
    Wrapper around requests.get with basic retries for transient errors.
    Retries on 429 and 5xx responses with incremental backoff (2s, 4s, 6s).
    Returns the first non-transient response; if all fail, returns the last response.
    """
    last = None
    for i in range(tries):
        r = requests.get(url, params=params, timeout=timeout)
        last = r
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 * (i + 1))
            continue
        return r
    assert last is not None
    return last  # likely an error; caller should inspect


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
    # allow raw Graph IDs as input:
    # - numeric post IDs: "1234567890"
    # - composite IDs:    "1234567890_987654321"
    if re.fullmatch(r"\d+(?:_\d+)?", url):
        return url
    return None


# ---------- Built-in demo comments (used by tests / demo) ----------
def load_test_comments() -> pd.DataFrame:
    # A tiny bilingual sample; keep your original long CSV if you prefer.
    data = StringIO(
        """comment_id,user_id,user_name,comment_text,created_time,like_count,love_count,haha_count,wow_count,sad_count,angry_count,care_count,reply_count,user_profile_link,user_gender,is_verified,language,parent_comment_id,permalink_url
cmt_1,user_1,নুসরাত জাহান শিলা,"খুব সুন্দর লিখেছেন, মুগ্ধ হলাম। 🙌",2024-08-20T10:05:00+0000,3,0,1,0,0,0,1,0,https://facebook.com/101001,Female,False,bn,,https://facebook.com/perm/1
cmt_2,user_2,John Abraham,"This is such an informative post. Thanks!",2024-08-20T10:06:00+0000,2,0,0,0,0,0,0,0,https://facebook.com/101002,Male,False,en,,https://facebook.com/perm/2
cmt_3,user_3,সাইফুল ইসলাম মেহেদী,"ভাই, এসব ফালতু কথা বাদ দেন 😅",2024-08-20T10:07:00+0000,1,0,2,0,0,0,0,0,https://facebook.com/101003,Male,False,bn,,https://facebook.com/perm/3
"""
    )
    df = pd.read_csv(data)
    return df


# ---------- Facebook fetcher (efficient) ----------
def _collect_typed_reactions_block() -> str:
    """
    Build a single fields string that asks Graph for per-type reaction summaries
    using field aliasing so we can parse counts from the same response.
    """
    types = ["LIKE", "LOVE", "HAHA", "WOW", "SAD", "ANGRY", "CARE"]
    parts = []
    for t in types:
        # reactions.type(LIKE).limit(0).summary(total_count).as(rx_like)
        parts.append(f"reactions.type({t}).limit(0).summary(total_count).as(rx_{t.lower()})")
    return ",".join(parts)


def _parse_reactions_from_obj(obj: Dict[str, Any]) -> Dict[str, int]:
    """Read counts from aliased reaction edges; fall back to zeros if absent."""
    out = {}
    for t in ["like", "love", "haha", "wow", "sad", "angry", "care"]:
        edge = obj.get(f"rx_{t}")
        count = 0
        if isinstance(edge, dict):
            summ = edge.get("summary") or {}
            count = int(summ.get("total_count") or 0)
        out[f"{t}_count"] = count
    return out


def fetch_comments_from_facebook(token: str, post_url: str) -> pd.DataFrame:
    """
    Fetches top-level comments for a given Facebook object URL/ID.
    Returns a DataFrame with:
      comment_id, user_id, user_name, comment_text, created_time,
      like_count, love_count, haha_count, wow_count, sad_count, angry_count, care_count,
      reply_count, user_profile_link, user_gender, is_verified, language, parent_comment_id, permalink_url
    """
    oid = extract_object_id(post_url)
    if not oid:
        print(f"[ERROR] Could not extract an object ID from: {post_url}")
        return pd.DataFrame()

    # Base request: bring back per-type reaction counts in one go (+permalink)
    fields = ",".join([
        "id",
        "message",
        "from",
        "created_time",
        "comment_count",
        "parent",
        "permalink_url",
        _collect_typed_reactions_block(),
    ])

    url = f"{GRAPH}/{oid}/comments"
    params = {
        "access_token": token,
        "fields": fields,
        "filter": "toplevel",  # change to 'stream' if you want replies too
        "limit": 100,
        "summary": "true",
    }

    all_rows: List[dict] = []
    page = 1

    while url:
        print(f"[INFO] Requesting page {page} of comments: {url}")
        r = _get(url, params=params, timeout=30)
        try:
            j = r.json()
        except Exception:
            print(f"[ERROR] Non-JSON response: {r.status_code} {r.text[:200]}")
            break

        data = j.get("data", [])
        for c in data:
            cid = c.get("id", "")
            u = c.get("from") or {}
            uid = u.get("id", "")
            uname = u.get("name", "")
            message = c.get("message", "") or ""

            # typed reaction counts (from the same payload)
            rx = _parse_reactions_from_obj(c)

            # language (best-effort, optional)
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
                    "reply_count": c.get("comment_count", 0) or 0,
                    "user_profile_link": "",     # kept for schema compatibility
                    "user_gender": "",            # Graph rarely returns gender
                    "is_verified": "",            # minimal user lookup for speed
                    "language": lang,
                    "parent_comment_id": (c.get("parent") or {}).get("id", ""),
                    "permalink_url": c.get("permalink_url", ""),
                }
            )

        # pagination
        paging = j.get("paging") or {}
        url = paging.get("next")
        params = None  # only first call needs params
        page += 1

    df = pd.DataFrame(all_rows)

    # If aliased reactions were blocked by API version/permissions, fall back:
    if not df.empty and "like_count" in df.columns and df["like_count"].sum() == 0:
        # Check whether all typed counts are zero AND there are comments; if so, try per-comment fallback
        print("[INFO] Reaction aliases may be unavailable; falling back to per-comment lookups (slower).")
        types = ["LIKE", "LOVE", "HAHA", "WOW", "SAD", "ANGRY", "CARE"]
        for idx, row in df.iterrows():
            cid = row["comment_id"]
            for t in types:
                try:
                    rr = _get(f"{GRAPH}/{cid}/reactions",
                              params={"type": t, "summary": "total_count", "limit": 0, "access_token": token},
                              timeout=20)
                    jj = rr.json()
                    cnt = int(((jj.get("summary") or {}).get("total_count")) or 0)
                except Exception:
                    cnt = 0
                df.at[idx, f"{t.lower()}_count"] = cnt

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

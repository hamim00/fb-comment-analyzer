import os
from flask import Flask, redirect, url_for, session, request, jsonify, render_template
import requests
from dotenv import load_dotenv

load_dotenv()

FB_APP_ID = os.getenv("FB_APP_ID")
FB_APP_SECRET = os.getenv("FB_APP_SECRET")
FB_REDIRECT_URI = os.getenv("FB_REDIRECT_URI")
SECRET_KEY = os.getenv("SECRET_KEY")

app = Flask(__name__)

# --- number formatting for UI (e.g., 1.2K, 3.4M) ---
def _abbr(n):
    try:
        n = int(n or 0)
    except Exception:
        return "0"
    for thresh, suf in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if n >= thresh:
            v = n / thresh
            s = f"{v:.1f}".rstrip("0").rstrip(".")
            return f"{s}{suf}"
    return str(n)

# register as a Jinja filter
app.jinja_env.filters["abbr"] = _abbr



app.secret_key = SECRET_KEY

# --------- HELPER FUNCTIONS ---------


def aggregate_totals(posts):
    def s(key):
        return sum(int(p.get(key) or 0) for p in posts)
    return {
        "posts": len(posts),
        "likes": s("like_count"),
        "comments": s("comment_count"),
        "shares": s("share_count"),
    }



def fetch_facebook_posts(token, page_id):
    url = f"https://graph.facebook.com/v19.0/{page_id}/posts"
    # reactions.type(LIKE) → like count; comments.summary → total comments; shares.count → shares
    fields = (
        "id,permalink_url,message,created_time,full_picture,"
        "attachments{media_type,media,url,subattachments},"
        "reactions.type(LIKE).limit(0).summary(total_count),"
        "comments.limit(0).summary(total_count),"
        "shares"
    )
    params = {"access_token": token, "fields": fields, "limit": 10}
    res = requests.get(url, params=params).json()

    posts = []
    for p in res.get("data", []):
        # media preview
        picture_url = p.get("full_picture")
        video_url = None
        for att in (p.get("attachments", {}) or {}).get("data", []) or []:
            mt = att.get("media_type")
            media = att.get("media", {}) or {}
            if mt == "video":
                video_url = att.get("url")
                picture_url = media.get("image", {}).get("src") or picture_url
            elif mt == "photo":
                picture_url = media.get("image", {}).get("src") or picture_url

        # counts
        like_count     = ((p.get("reactions") or {}).get("summary") or {}).get("total_count", 0)
        comment_count  = ((p.get("comments")  or {}).get("summary") or {}).get("total_count", 0)
        share_count    =      (p.get("shares") or {}).get("count", 0)

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

# --------- ROUTES ---------

@app.route('/')
def index():
    user = session.get('user')
    if not user:
        return render_template('no_login.html')
    # If logged in, always redirect to posts page now!
    return redirect(url_for('show_posts'))

@app.route('/login')
def login():
    fb_auth_url = (
        f"https://www.facebook.com/v19.0/dialog/oauth"
        f"?client_id={FB_APP_ID}"
        f"&redirect_uri={FB_REDIRECT_URI}"
        f"&scope=public_profile,pages_read_engagement,pages_read_user_content"
        f"&response_type=code"
    )
    return redirect(fb_auth_url)

@app.route('/facebook/callback')
def facebook_callback():
    code = request.args.get('code')
    token_url = "https://graph.facebook.com/v19.0/oauth/access_token"
    token_params = {
        'client_id': FB_APP_ID,
        'redirect_uri': FB_REDIRECT_URI,
        'client_secret': FB_APP_SECRET,
        'code': code
    }
    token_response = requests.get(token_url, params=token_params).json()
    access_token = token_response.get('access_token')
    if not access_token:
        return "Failed to retrieve access token. Response: " + str(token_response), 400

    # Fetch user info
    user_info = requests.get("https://graph.facebook.com/v19.0/me", params={
        'fields': 'id,name,picture',
        'access_token': access_token
    }).json()
    session['user'] = user_info
    session['user_token'] = access_token  # store user token

    # Fetch pages user manages
    pages_url = "https://graph.facebook.com/v19.0/me/accounts"
    pages_res = requests.get(pages_url, params={'access_token': access_token}).json()
    pages = pages_res.get("data", [])
    session['pages'] = pages  # Save the list of pages

    # Redirect to account chooser
    return redirect(url_for('choose_account'))

@app.route('/choose_account', methods=['GET', 'POST'])
def choose_account():
    user = session.get('user')
    pages = session.get('pages', [])

    if request.method == 'POST':
        selected_type = request.form.get('account_type')
        selected_page_id = request.form.get('page_id')
        if selected_type == 'user':
            session['access_token'] = session['user_token']
            session['account_type'] = 'user'
            session['account_name'] = user.get('name')
            session.pop('page_id', None)
        elif selected_type == 'page' and selected_page_id:
            for page in pages:
                if page['id'] == selected_page_id:
                    session['access_token'] = page['access_token']
                    session['account_type'] = 'page'
                    session['account_name'] = page['name']
                    session['page_id'] = page['id']
                    break
        # After account chosen, go to post selector
        return redirect(url_for('show_posts'))

    return render_template('choose_acc.html', user=user, pages=pages)

@app.route('/show_posts')
def show_posts():
    token = session.get('access_token')
    account_type = session.get('account_type')
    account_name = session.get('account_name')
    user = session.get('user')

    if account_type == "user":
        page_id = user['id']
    else:
        page_id = session.get('page_id')

    posts = fetch_facebook_posts(token, page_id) if (token and page_id) else []
    # Pass token to template for Analyze button
    return render_template(
        'posts.html', 
        posts=posts, 
        account_name=account_name, 
        account_type=account_type, 
        access_token=token
    )


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# Your fetch_comments_api and any other backend logic remain here for Streamlit/other use.

if __name__ == "__main__":
    app.run(debug=True)

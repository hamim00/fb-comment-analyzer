# app.py
from backend import app as application
from blueprints.analysis import analysis_bp
application.register_blueprint(analysis_bp, url_prefix="/analyze")

if __name__ == "__main__":
    application.run(debug=True)

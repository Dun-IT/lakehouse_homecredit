from flask import Flask, render_template_string, request, redirect, url_for, session, flash, send_file
import pandas as pd
import json
import os
import io
import joblib
import io
import uuid
from catboost import CatBoostClassifier
from jinja2 import DictLoader
from functools import wraps

app = Flask(__name__)
app.secret_key = os.urandom(24)

# In-memory store for predictions (server-side)
PREDICTION_STORE = {}

# Load user credentials
def load_users(path='users.json'):
    default = {'users': {'admin': 'admin'}}
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump(default, f, indent=2)
        return default['users']
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get('users', default['users'])
    except json.JSONDecodeError:
        with open(path, 'w') as f:
            json.dump(default, f, indent=2)
        return default['users']

USERS = load_users()
# MinIO settings
BUCKET = 'lakehouse'
CSV_PATH = 'raw/application_test.csv'

def get_minio_opts():
    return {
        'key': 'minio',
        'secret': 'minio_admin',
        'client_kwargs': {'endpoint_url': 'http://localhost:9000'}
    }

# Load models once
def load_models():
    base = os.path.join(os.getcwd(), 'models')
    cat = CatBoostClassifier()
    cat.load_model(os.path.join(base, 'catboost_model.cbm'))
    sgd = joblib.load(os.path.join(base, 'hcdr_sgdlogistic_imputed.pkl'))
    lgbm = joblib.load(os.path.join(base, 'lgbm_model.pkl'))
    return {
        'CatBoost - Score: 76,68': cat,
        'SGDLogistic - Score: 73,55': sgd,
        'LightGBM - Score: 76,59': lgbm
    }
MODELS = load_models()


# Templates
LOGIN_PAGE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css">
  <title>Login - Home Credit Default Risk</title>
</head>
<body class="d-flex justify-content-center align-items-center" style="height:100vh;background:#f8f9fa;">
  <form method="post" style="background:#fff;padding:2rem;border-radius:.5rem;box-shadow:0 4px 12px rgba(0,0,0,0.1);width:320px;">
    <h4 class="mb-4 text-center text-primary">Login</h4>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="alert alert-danger">{{ messages[0] }}</div>
      {% endif %}
    {% endwith %}
    <div class="form-group"><label>Username</label><input name="username" class="form-control" required></div>
    <div class="form-group"><label>Password</label><input type="password" name="password" class="form-control" required></div>
    <button class="btn btn-primary btn-block">Login</button>
  </form>
</body>
</html>
'''

BASE_LAYOUT = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css">
  <title>{{ title }} - Home Credit DR</title>
  <style>
    body { display:flex; margin:0; font-family:Arial, sans-serif; background:#e9ecef; }
    .sidebar { display:flex; flex-direction:column; justify-content:space-between; flex:0 0 240px; background:#ffffff; border-right:1px solid #dee2e6; min-height:100vh; height: 100vh;}
    .sidebar .header { padding:1rem; background:#0d6efd; color:#fff; font-size:1.25rem; text-align:center; }
    .nav-links { padding:1rem; }
    .nav-link { display:block; margin-bottom:0.5rem; color:#212529; font-weight:500; padding:0.5rem 1rem; border-radius:0.25rem; text-decoration:none; }
    .nav-link.active { background:#0d6efd; color:#fff; }
    .nav-link:hover { background:#f1f3f5; color:#0d6efd; }
    .content { flex:1; padding:1.5rem; background:#ffffff; height: 100vh; overflow-x:hidden; overflow-y: auto; }
    .table th, .table td { padding:.5rem; white-space:nowrap; }
    .table-wrapper { width:100%; max-width:100%; max-height:70vh; overflow-x:auto; overflow-y:auto; background:#fff; border:1px solid #dee2e6; border-radius:.25rem; box-shadow:0 2px 4px rgba(0,0,0,0.05); }
    .table-wrapper table { width:max-content; }
    .btn-logout { background:#dc3545; color:#fff; border:none; }
    .btn-logout:hover { background:#c82333; }
  </style>
</head>
<body>
  <div class="sidebar">
    <div>
      <div class="header">Home Credit Default Risk</div>
      <div class="nav-links">
        <a class="nav-link {% if active=='customers' %}active{% endif %}" href="{{ url_for('customer_list') }}">Customer</a>
        <a class="nav-link {% if active=='predict' %}active{% endif %}" href="{{ url_for('predict_page') }}">Predict</a>
      </div>
    </div>
    <div class="p-3">
      <a class="btn btn-logout btn-block" href="{{ url_for('logout') }}">Logout</a>
    </div>
  </div>
  <div class="content">
    <h2 class="mb-4 text-primary">{{ title }}</h2>
    {% block body %}{% endblock %}
  </div>
</body>
</html>
'''

# Setup templates loader
app.jinja_loader = DictLoader({'login.html': LOGIN_PAGE, 'layout.html': BASE_LAYOUT})

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('user'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# Routes
@app.route('/login', methods=['GET', 'POST'])

def login():
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']
        if USERS.get(u) == p:
            session['user'] = u
            return redirect(url_for('customer_list'))
        flash('Invalid credentials')
    return render_template_string('{% include "login.html" %}')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/customers')
@login_required
def customer_list():
    limit = request.args.get('limit', 20, type=int)
    df = pd.read_csv(f's3://{BUCKET}/{CSV_PATH}', storage_options=get_minio_opts(), nrows=limit)
    table = df.to_html(classes="table table-hover table-sm mb-0", index=False, border=0)
    return render_template_string(
        '{% extends "layout.html" %}{% block body %}' +
        '<div class="form-inline mb-3"><label class="mr-2">Rows:</label>' +
        '<select id="limitSelect" class="form-control form-control-sm mr-2">' +
        ''.join([f'<option value="{n}" {{% if {n}==limit %}}selected{{% endif %}}>{n}</option>' for n in [10,20,50,100,200]]) +
        '</select></div>' +
        '<div class="table-wrapper">'+table+'</div>' +
        '<script>document.getElementById("limitSelect").addEventListener("change",function(){window.location.href=`/customers?limit=${this.value}`;});</script>' +
        '{% endblock %}',
        title='Customer List', active='customers', limit=limit
    )

# Predict route
@app.route('/predict', methods=['GET','POST'])
@login_required
def predict_page():
    error = None
    result_html = ''
    download_key = None
    # handle limit param separately
    if request.method == 'POST':
        model_name = request.form.get('model')
        file = request.files.get('file')
        limit = int(request.form.get('limit', 20))
        if model_name and file:
            try:
                df_in = pd.read_csv(file)
                model = MODELS.get(model_name)
                # Align features for sklearn models
                if hasattr(model, 'feature_names_in_'):
                    cols = list(model.feature_names_in_)
                    df_in = df_in.reindex(columns=cols, fill_value=0)
                # Predict
                if hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(df_in)[:,1]
                else:
                    preds = model.predict(df_in)
                df_in['prediction_score'] = preds
                # slice for display
                df_display = df_in.head(limit)
                result_html = df_display.to_html(classes="table table-striped table-sm", index=False)
                # prepare download CSV with only ID and score
                out_df = df_in[['SK_ID_CURR', 'prediction_score']]
                buf = io.StringIO()
                out_df.to_csv(buf, index=False)
                download_key = str(uuid.uuid4())
                PREDICTION_STORE[download_key] = buf.getvalue()
            except Exception as e:
                error = str(e)
        else:
            error = 'Please select model and CSV file.'
    # Build form
    form = '<form method="post" enctype="multipart/form-data">'
    form += '<div class="form-group"><label>Choose Model</label>'
    form += '<select name="model" class="form-control mb-3">'
    for m in MODELS.keys(): form += f'<option value="{m}">{m}</option>'
    form += '</select></div>'
    form += '<div class="form-group"><label>Upload CSV</label>'
    form += '<input type="file" name="file" accept=".csv" class="form-control-file mb-3"></div>'
    form += '<div class="form-group"><label>Rows to display</label>'
    form += '<input type="number" name="limit" value="20" min="1" class="form-control mb-3"/></div>'
    form += '<button type="submit" class="btn btn-success">Predict</button>'
    form += '</form>'
    body = ''
    if error: body += f'<div class="alert alert-danger">Error: {error}</div>'
    body += form
    if result_html and download_key:
        body += '<h4 class="mt-4">Prediction Results</h4>' + '<div class="table-wrapper">' + result_html + '</div>'
        body += f'<a href="/download/{download_key}" class="btn btn-primary mt-2">Download Results</a>'
    return render_template_string(
        '{% extends "layout.html" %}{% block body %}' + body + '{% endblock %}',
        title='Predict', active='predict'
    )

@app.route('/download/<key>')
@login_required
def download_predictions(key):
    csv_str = PREDICTION_STORE.pop(key, '')
    return send_file(
        io.BytesIO(csv_str.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='predictions.csv'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

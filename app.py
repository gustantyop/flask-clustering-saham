from flask import Flask, request, render_template
from model_logic import run_clustering_with_plot
import os

app = Flask(__name__)
DATA_FOLDER = "data"

@app.route('/')
def home():
    filenames = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    return render_template('index.html', filenames=filenames)

@app.route('/cluster', methods=['POST'])
def cluster():
    selected_files = request.form.getlist('files')
    result = run_clustering_with_plot(selected_files, data_path=DATA_FOLDER)

    if "error" in result:
        return f"<h3>{result['error']}</h3><br><a href='/'>Kembali</a>"

    return render_template("result.html",
                           inertia=result['inertia'],
                           proportions=result['proportions'],
                           allocation=result['allocation'])

if __name__ == '__main__':
    app.run(debug=True)
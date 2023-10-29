from flask import Flask, render_template, request
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import os
app = Flask(__name__)
    
# # Чтение данных из CSV файла
# data = pd.read_csv('test.csv')

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def save_and_rename_file(file):
    # Сохраняем файл в папку uploads
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'tovar.csv')
    file.save(file_path)
    data = pd.read_csv(file_path)

def save_result(file):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'tovar.csv')
    file.save(file_path)
    data = pd.read_csv(file_path)


@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['file']
    if str(file) == """<FileStorage: '' ('application/octet-stream')>""":
        # print('нечего загружать')
        pass
    else:
        save_and_rename_file(file)
        # print('save_and_rename_file()')
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'tovar.csv')
        data = pd.read_csv(file_path)
        return render_template('page2.html', data=data)
    return render_template('index.html')



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'tovar.csv')
    data = pd.read_csv(file_path)
    # print(query)
    if query:
        filtered_data = data[data['Наименование продукции'].apply(lambda x: fuzz.partial_ratio(x, query)) > 60]
    else:
        filtered_data = data

    return render_template('page3.html', data=filtered_data)


    print('я работаю')


@app.route('/result', methods=['POST'])
def result():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'tovar.csv')
    data = pd.read_csv(file_path)

    # Добавление новой строки в начало DataFrame
    new_row_data = {'Наименование продукции': 'Новая запись', 'Группа продукции': 'Новая запись','Требуемое оборудование':'Новая запись','Рекомендации':'Новая запись'}  # Замените 'Column1', 'Column2', 'Value1', 'Value2' на нужные данные
    data.loc[-1] = new_row_data
    data.index = data.index + 1
    data = data.sort_index()

    # Сохранение DataFrame в новый CSV файл
    result_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'tovar.csv')
    data.to_csv(result_file_path, index=False)


    data = pd.read_csv(file_path)

    return render_template('page3.html', data=data)


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import os
from transformers import AutoModel, AutoTokenizer
from flask import send_file
from utils import * 
import re

app = Flask(__name__)
    
# # Чтение данных из CSV файла
# data = pd.read_csv('test.csv')

tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_mt_nlu_ru")
model = AutoModel.from_pretrained("ai-forever/sbert_large_mt_nlu_ru")
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

@app.route('/download', methods=['POST'])
def down():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'tovar.csv')
    return send_file(file_path, as_attachment=True)


@app.route('/result', methods=['POST'])
def result():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'tovar.csv')
    dataset = pd.read_csv(file_path)

    recomedations = []
    column_group_47 = []
    column_group_18 = []
    column_group_5 = []

    for _, row in dataset.iterrows():
        name_product = row['Группа продукции']
        result = standarts(name_product, tokenizer, model)
        new_standarts, group_18, group_5 = result
        
        
        extracted_titles = []
        for file_name in new_standarts:
            match = re.search(r'((?:ГОСТ|МУ|СТ РК ИСО|СТБ ISO|СТБ ГОСТ|СанПиН|МУК|РД|ПНД Ф|ГОСТ Р ИСО|ГОСТ Р|МВИ\.МН)[\s.-]+[0-9.-]+)', file_name)
            if match:
                extracted_titles.append(match.group(1))
        
        tools_equipments = []
        
        for extracted_title in extracted_titles:
            f_eqp = []
            finded = False
            with open('equipment.txt', mode='r', encoding='utf-8') as file:
                for line in file:
                    if finded == True:
                        if '}' in line:
                            break
                        f_eqp.append(line)
                
                    if extracted_title in line and finded == False:
                        finded = True
            for line in f_eqp:
                if line != '\n':
                    tools_equipments.append(line.replace('\n', '').strip())

        rec = '; '.join(tools_equipments)
        recomedations.append(rec)
        column_group_18.append(group_18)
        #column_group_47.append(group_47)
        column_group_5.append(group_5)



    dataset['Рекомендации по ТО'] = recomedations
    dataset['Верно/Неверно'] = dataset.apply(lambda row: calculate_cosine_similarity(row['Рекомендации по ТО'], row['Техническое оборудование']), axis=1)
    dataset['Верно/Неверно'] = dataset['Верно/Неверно'].apply(lambda x: 'Верно' if x > 0.24 else 'Неверно')

    result_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'tovar.csv')
    dataset.to_csv(result_file_path, index=False)

    return render_template('page3.html', data=dataset)


if __name__ == '__main__':
    app.run(debug=True)

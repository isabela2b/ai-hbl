import os, shutil, json, glob, re, requests, io, cv2 #keras
import pandas as pd
import numpy as np
import pymssql
from pymssql import _mssql
from pdf2image import convert_from_bytes
from PyPDF2 import PdfFileMerger
from PIL import Image

from functions import * 

from azure.core.exceptions import ResourceNotFoundError
from azure.ai.formrecognizer import DocumentAnalysisClient, AnalyzeResult
from azure.core.credentials import AzureKeyCredential

endpoint = "https://ai-cargomation.cognitiveservices.azure.com/"
credential = AzureKeyCredential("a6a3fb5f929541648c788d45e6566603")
document_analysis_client = DocumentAnalysisClient(endpoint, credential)
default_model_id = "hbl1"
data_folder = "../ai-data/test-ftp-folder/"
#data_folder = "E:/A2BFREIGHT_MANAGER/"
poppler_path = r"C:\Program Files\poppler-21.03.0\Library\bin"

def container_separate(containers):
    """
    This function receives a string containing all containers and outputs them into a list
    of containers following standardized formatting.
    """
    pattern = "[a-zA-Z]{4}[0-9]{7}"
    formatted_container = re.findall(pattern, re.sub('[^A-Za-z0-9]+','', containers))
    return formatted_container

def filename_filter(filename):
    """
    This function receives a file name string, usually an invoice number, and removes special characters.
    """
    return re.sub('[^A-Za-z0-9]+', '', filename)

def form_recognizer_one(document, file_name, page_num, model_id=default_model_id):
    prediction={}

    table = {'container_number': [], 'seal': [], 'container_type':[], 'chargeable_weight':[], 'volume': [], 'package_count': []}
    poller = document_analysis_client.begin_analyze_document(model=model_id, document=document)
    result = poller.result()

    for analyzed_document in result.documents:
        print("Document was analyzed by model with ID {}".format(result.model_id))
        print("Document has confidence {}".format(analyzed_document.confidence))
        for name, field in analyzed_document.fields.items():
            if name=='table':
                #print("Field '{}' ".format(name))
                for row in field.value:
                    row_content = row.value
                    for key, item in row_content.items():
                        print('Field {} has value {}'.format(key, item.value))
                        if key == "container_number" and item.value:
                            table[key].append(container_separate(item.value))
                        else:
                            table[key].append(item.value)
            else:
                prediction[name]=field.value
                print("Field '{}' has value '{}' with confidence of {}".format(name, field.value, field.confidence))

    if prediction['container_number'] is not None:
        prediction['container_number'] = container_separate(prediction['container_number'])
    prediction['table'] = table
    prediction['filename'] = file_name
    prediction['page'] = page_num

    return prediction

def multipage_combine(prediction_mult, shared_invoice, pdf_merge = False):
    """
    Receives json file of predictions and dict of shared invoices and 
    combines all pages with the same invoice number.
    """
    try:
        res = {}
        for i, v in shared_invoice.items():
            res[v] = [i] if v not in res.keys() else res[v] + [i]
        merged_predictions = {}
        print(res)
        for invoice_num, pages in res.items():
            page_nums = []
            #page has the file name, so what we can do is get the file from the split folder
            output = PdfFileMerger()
            new_file_name = filename_filter(invoice_num) + "." +file_ext(pages[0])
            for idx, page in enumerate(pages):
                if idx==0:
                    merged_predictions[new_file_name] = prediction_mult[page].copy()
                    page_nums.append(prediction_mult[page]['page'])
                else:
                    for field, item_value in prediction_mult[page].items():
                        if field == 'table':
                            for table_key, table_value in item_value.items():
                                merged_predictions[new_file_name]['table'][table_key].extend(table_value)
                        elif field == 'page':
                            page_nums.append(item_value)
                        elif merged_predictions[new_file_name][field] is None and item_value is not None:
                            merged_predictions[new_file_name][field] = item_value
                split_file_path = data_folder+"SPLIT/"+page
                output.append(split_file_path)
            merged_predictions[new_file_name]['page'] = page_nums
            output.write(data_folder+"SPLIT/"+new_file_name)
            output.close()
        return merged_predictions
    except Exception as ex:
        return str(ex)

def query_webservice_user(webservice_user):
    #query the server
    conn = pymssql.connect('a2bserver.database.windows.net', 'A2B_Admin', 'v9jn9cQ9dF7W', 'a2bcargomation_db')
    cursor = conn.cursor(as_dict=True)
    cursor.execute("SELECT TOP (1) * FROM [dbo].[user_webservice] WHERE [user_id] = %s", webservice_user)
    user_query=cursor.fetchone()
    cursor.close()
    return user_query

def add_webservice_user(predictions, file, user_query):
    predictions[file]["webservice_link"] = user_query['webservice_link'] #"https://a2btrnservices.wisegrid.net/eAdaptor"  
    predictions[file]["webservice_username"] =  user_query['webservice_username']#"A2B"
    predictions[file]["webservice_password"] = user_query['webservice_password']#"Hw7m3XhS"
    predictions[file]["server_id"] = user_query['server_id']# "TRN"
    predictions[file]["enterprise_id"] =  user_query['enterprise_id'] #"A2B"
    predictions[file]["company_code"] = user_query['company_code']#"SYD"
    return predictions

def push_parsed_inv(predictions, process_id, user_id):
    conn = pymssql.connect('a2bserver.database.windows.net', 'A2B_Admin', 'v9jn9cQ9dF7W', 'a2bcargomation_db')
    cursor = conn.cursor(as_dict=True)
    cursor.execute("UPDATE [dbo].[match_apinvoice] SET [parsed_inv]=%s WHERE [process_id]=%s AND [user_id]=%s", (predictions, process_id, user_id))
    conn.commit()


def extract_compare(file_bytes, filename, user_id, process_id):
    #url = "https://cargomation.com:5200/redis/apinvoice/compare"
    predictions = {}
    shared_invoice = {} #page and invoice number
    user_query = query_webservice_user(user_id)
    ext = file_ext(filename)

    if ext == "pdf":
        images = convert_from_bytes(file_bytes, grayscale=True, fmt="jpeg") #, poppler_path=poppler_path
        inputpdf = PdfFileReader(io.BytesIO(file_bytes))
        if inputpdf.isEncrypted:
            try:
                inputpdf.decrypt('')
                print('File Decrypted (PyPDF2)')
            except:
                print("Decryption error")
        #classify and split
        for page, image in enumerate(images):
            print(type(image))
            if hbl_page(image) == 1:
                #pred = classify_carrier(image)
                output = PdfFileWriter()
                output.addPage(inputpdf.getPage(page)) #pages begin at zero in pdffilewriter
                page_num = page+1 #for counters to begin at 1
                split_file_name = file_name(filename) +"_pg"+str(page_num)+".pdf"
                split_file_path = data_folder+"SPLIT/"+split_file_name
                with open(split_file_path, "wb") as outputStream:
                    output.write(outputStream) #this can be moved to only save when the split file is AP
                fd = open(split_file_path, "rb")
                predictions[split_file_name] = form_recognizer_one(document=fd.read(), file_name=filename, page_num=page_num)
                shared_invoice[split_file_name] = predictions[split_file_name]['hbl_number']

        predictions = multipage_combine(predictions, shared_invoice)

    elif ext in ["jpg", "jpeg", "png",'.bmp','.tiff']:
        pil_image = Image.open(io.BytesIO(file_bytes)).convert('L').convert('RGB') 
        if hbl_page(pil_image) == 1:
            predictions[filename] = form_recognizer_one(document=file_bytes, file_name=filename, page_num=1)
    else:
        return "File type not allowed."

    print(predictions)
    for file in predictions:
        predictions = add_webservice_user(predictions, file, user_query)
        predictions[file]['process_id'] = process_id 
        predictions[file]['merged_file_path'] = os.path.abspath(data_folder+"SPLIT/"+file)
        payload = {
            "user_id": user_id,
            "jsonstring": json.dumps(predictions[file])
            }
        #r = requests.post(url, auth=('admin', r'u\}M[6zzAU@w8YLx'), headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        #print(r.status_code)
    #push_parsed_inv(json.dumps(predictions), process_id, user_id)
        y = json.dumps(payload, indent=4)
        with open(data_folder+'PREDICTIONS/'+file_name(file)+'.json', 'w') as outfile:
            outfile.write(y)

    return payload
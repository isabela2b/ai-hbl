import os, shutil, json, glob, re, io, cv2, requests #keras
import pandas as pd
import numpy as np
import pymssql
from pymssql import _mssql
from pdf2image import convert_from_bytes
from PyPDF2 import PdfWriter, PdfReader, PdfMerger

from PIL import Image
from collections import defaultdict

from functions import * 

from azure.core.exceptions import ResourceNotFoundError
from azure.ai.formrecognizer import DocumentAnalysisClient, AnalyzeResult
from azure.core.credentials import AzureKeyCredential

endpoint = "https://ai-cargomation.cognitiveservices.azure.com/"
credential = AzureKeyCredential("a6a3fb5f929541648c788d45e6566603")
document_analysis_client = DocumentAnalysisClient(endpoint, credential)
default_model_id = "hbl1_10"
data_folder = "../ai-data/test-ftp-folder/"
#data_folder = "E:/A2BFREIGHT_MANAGER/"
#poppler_path = r"C:\Program Files\poppler-21.03.0\Library\bin"

class_indices = {'hbl': 0, 'mbl': 1, 'others': 2}
#mbl_carriers_indices = {0: 'anl', 1: 'anl-2', 2: 'carotrans', 3: 'cmacgm', 4: 'cmacgm-2', 5: 'cosco', 6: 'cosco-2', 7: 'direct', 8: 'evergreen', 9:'evergreen-2', 10: 'goldstar', 11: 'goldstar-2', 12: 'hamsud', 13: 'hapllo', 14: 'happlo-2', 15: 'hmm', 16: 'hmm-2', 17: 'maersk', 18: 'maersk-2', 19: 'mariana', 20: 'msc', 21: 'msc-2', 22: 'ocenet', 23: 'ocenet-2', 24: 'oocl', 25: 'oocl-2', 26: 'other', 27: 'pil', 28: 'sinotrans', 29: 'tslines', 30: 'tslines-2', 31: 'yangming'}
mbl_carriers_match = {0: 'anl', 1: 'anl-2', 2: 'carotrans', 3: 'cmacgm', 4: 'cmacgm-2', 5:'mbl_cosco_17', 6:'mbl_attached_4', 7: 'direct', 8: 'mbl_evergreen_3', 9:'evergreen-2', 10: 'mbl_goldstar_3', 11: 'mbl_goldstar2', 12: 'hamsud', 13: 'hapllo', 14: 'happlo-2', 15: 'mbl_hmm_2', 16: 'hmm-2', 17: 'maersk', 18: 'maersk-2', 19: 'mbl_mariana_3', 20: 'mbl_msc_1', 21: 'mbl_msc2', 22: 'ocenet', 23: 'oocl', 24: 'oocl-2', 25: 'other', 26: 'pil', 27: 'sinotrans', 28: 'mbl_tslines_5', 29: 'tslines-2', 30: 'yangming'}
mbl_carriers = {0: 'ANL', 1: 'ANL', 2: 'Carotrans', 3: 'Cmacgm', 4: 'Cmacgm-2', 5:'COSCO SHIPPING LINES CO ., LTD.', 6:'COSCO SHIPPING LINES CO ., LTD.', 7: 'Direct Shipping', 8: 'Evergreen Line', 9:'Evergreen Line', 10: 'Gold Star Line Ltd.', 11: 'Gold Star Line Ltd.', 12: 'Hamburg Sud', 13: 'Hapag Lloyd', 14: 'Hapag Lloyd', 15: 'HMM', 16: 'HMM', 17: 'Maersk', 18: 'Maersk', 19: 'Mariana Express Lines Pte Ltd', 20: 'MEDITERRANEAN SHIPPING COMPANY S.A.', 21: 'MEDITERRANEAN SHIPPING COMPANY S.A.', 22: 'Ocean Network Express', 23: 'OOCL', 24: 'OOCL', 25: None, 26: 'PIL', 27: 'Sinotrans', 28: 'T.S. LINES', 29: 'T.S. LINES', 30: 'Yangming'}
hbl_carriers_match = {0: 'attached', 1: 'hbl_hls_7', 2: 'other', 3: 'hbl_sinotrans_3', 4: 'sinotrans-ver2'}


def container_separate(containers):
    """
    This function receives a string containing all containers and outputs them into a list
    of containers following standardized formatting.
    """
    if containers:
        pattern = "[a-zA-Z]{4}[0-9]{7}"
        formatted_container = re.findall(pattern, re.sub('[^A-Za-z0-9]+','', containers))
        return formatted_container
    else:
        return None

def special_char_filter(filename):
    """
    This function receives a file name string, usually an invoice number, and removes special characters.
    """
    return re.sub('[^A-Za-z0-9]+', '', filename)

def mbl_num_filter(mbl_number): #fix mbl filter, remove everything before the special character
    new_mbl = re.findall("[a-zA-Z]{4}[0-9]{10}", special_char_filter(mbl_number))
    return new_mbl[0]

def mbl_filter(prediction):
    prediction['doc_type'] = "MBL"

    if prediction.__contains__('vessel_voyage'):
        for substring in prediction['vessel_voyage'].split(" "):
            if re.search(r'\d', substring):
                prediction['voyage_number'] = substring
        prediction['vessel_name'] = prediction['vessel_voyage'].replace(prediction['voyage_number'], "").strip()

    return prediction

def hbl_filter(prediction):
    prediction['doc_type'] = "HBL"
    return prediction

def gen_table_filter(table):
    table['container_number'] = [container_separate(container)[0] for container in table['container_number']]
    return table

def table_filter(table):
    try:
        table['container_number'] = [container_separate(container)[0] for container in table['container_number']]
        table['container_type'] = [container_type_filter(container) for container in table['container_type']]
        table['chargeable_weight'] = [re.sub('[^\w. ]+', '', container) for container in table['chargeable_weight']]
    except:
        pass
    return table

def find_release_type(surrendered, telex_release, ebl, number_original):
    release_type = "OBR"
    if surrendered or telex_release or ebl or (number_original and any(map(number_original.lower().__contains__, ["one", "zero", "1", "0", "none"]))):
        release_type = "EBL"
    return release_type

def container_type_filter(container_type):
    return re.findall("[0-9]{2}\s[a-zA-Z]{2}", container_type)[0]
    
def separate_package(table):
    for idx,container_type in enumerate(table['container_type']):
        if container_type:
            table['container_type'][idx] = re.findall("[0-9]{2}[a-zA-Z]{2}",container_type)[0]
            remaining = container_type.replace(table['container_type'][idx], '').split(';')
            if len(remaining)>=2:
                table['chargeable_weight'][idx] = special_char_filter(remaining[0])
                table['volume'][idx] = special_char_filter(remaining[1])
    return table

def table_row_filter(table):
    formatted_table = {'container_number': [], 'seal': [], 'container_type':[], 'chargeable_weight':[], 'volume': [], 'package_count': []}
    for row in table['row']:
        try:
            row = row.split('/')
            formatted_table['container_number'].append(container_separate(row[0])[0])
            formatted_table['seal'].append(special_char_filter(row[1]))
            formatted_table['container_type'].append(special_char_filter(row[2]))
            formatted_table['package_count'].append(row[3])
            formatted_table['chargeable_weight'].append(row[4])
            formatted_table['volume'].append(row[5])
        except:
            pass
    return formatted_table

def evergreen_table_filter(table):
    for idx, row in enumerate(table['container_number']):
        new_container = container_separate(row)
        if new_container: #and table['container_number'][idx]
            row = row.split('/')
            table['container_number'][idx] = new_container[0]
            table['container_type'][idx] = special_char_filter(row[1])
            table['seal'][idx] = special_char_filter(row[2])
            if len(row) > 3:
                table['package_count'][idx] = row[3]
        else:
            table['container_number'][idx] = None
    return table

def tslines_table_filter(table):
    for idx, row in enumerate(table['container_number']):
        new_container = container_separate(row)
        if new_container: #and table['container_number'][idx]
            row = row.split('/')
            table['container_number'][idx] = new_container[0]
            table['container_type'][idx] = special_char_filter(row[1])
        else:
            table['container_number'][idx] = None
    return table

def goldstar_table_filter(table):
    for idx, row in enumerate(table['container_number']):
        new_container = container_separate(row)
        if new_container: #and table['container_number'][idx]
            table['container_number'][idx] = new_container[0]
            row = table['container_type'][idx].split('/')
            table['seal'][idx] = row[0].split(':')[1]
            table['container_type'][idx] = special_char_filter(row[1])
        else:
            table['container_number'][idx] = None
    return table

def table_remove_null(table):
    indexes = sorted([i for i,x in enumerate(table['container_number']) if x is None], reverse=True)
    for col in table:
        for index in indexes:
            del table[col][index]
    return table

def form_recognizer_one(document, file_name, page_num, carrier, model_id=default_model_id):
    prediction={}
    table = defaultdict(list)
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
                        if key == "seal" and item.value:
                            table[key].append(special_char_filter(item.value))
                        else:
                            table[key].append(item.value)
            elif name=='carrier': #and not field.value
                prediction[name]=carrier
            else:
                prediction[name]=field.value
                print("Field '{}' has value '{}' with confidence of {}".format(name, field.value, field.confidence))

    if prediction.__contains__('container_number') and prediction['container_number'] is not None:
        prediction['container_number'] = container_separate(prediction['container_number'])

    try:
        prediction['table'] = separate_package(table)
    except:
        prediction['table'] = table

    if prediction.__contains__('incoterm') and prediction['incoterm']:        
        if "freight" in prediction['incoterm'].lower():
            prediction['incoterm'] = re.sub('[^A-Za-z ]+', '', prediction['incoterm'])
        else:
            prediction['incoterm'] = None

    prediction['filename'] = file_name
    prediction['page'] = page_num

    if prediction.__contains__('consignee_address') and prediction['consignee_address']:
        for substring in prediction['consignee_address'].split(" "):
            if re.search('[a-zA-Z]{3}', substring) and len(substring) == 3 and substring.upper() in ["NSW", "QLD", "VIC", "TAS", "ACT"]:
                prediction['state_code'] = substring
                break

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
            #output = PdfFileMerger()
            new_file_name = special_char_filter(invoice_num) + "." +file_ext(pages[0])
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
                #split_file_path = data_folder+"SPLIT/"+page
                #output.append(split_file_path)
            merged_predictions[new_file_name]['page'] = page_nums
            #output.write(data_folder+"SPLIT/"+new_file_name)
            #output.close()
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
    cursor.execute("UPDATE [dbo].[match_registration] SET [parsed_input]=%s WHERE [process_id]=%s AND [user_id]=%s", (predictions, process_id, user_id))
    conn.commit()

def mbl_carrier_model(image):
    return mbl_carriers_match[classify_mbl_carrier(image)]

def predict(file_bytes, filename, process_id, user_id):
    predictions = {}
    shared_invoice = {}
    payload = {}
    hbl_list = []

    user_query = query_webservice_user(user_id)
    ext = file_ext(filename)

    if ext == "pdf":
        images = convert_from_bytes(file_bytes, grayscale=True, fmt="jpeg") #, poppler_path=poppler_path
        inputpdf = PdfReader(io.BytesIO(file_bytes), strict=False)
        if inputpdf.is_encrypted:
            try:
                inputpdf.decrypt('')
                print('File Decrypted (PyPDF2)')
            except:
                print("Decryption error")
        
        #classify and split
        for page, image in enumerate(images):
            pred = classify_page(image)
            if pred == class_indices['hbl']:
                #if hbl_page(image) == 1:
                output = PdfWriter()
                output.add_page(inputpdf.pages[page]) #pages begin at zero in pdffilewriter
                page_num = page+1 #for counters to begin at 1
                split_file_name = file_name(filename) +"_pg"+str(page_num)+".pdf"
                split_file_path = data_folder+"SPLIT/"+split_file_name
                with open(split_file_path, "wb") as outputStream:
                    output.write(outputStream) #this can be moved to only save when the split file is AP
                fd = open(split_file_path, "rb")
                carrier = classify_hbl_carrier(image)
                predictions[split_file_name] = form_recognizer_one(document=fd.read(), file_name=filename, page_num=page_num, model_id=hbl_carriers_match[carrier], carrier=mbl_carriers[carrier])
                predictions[split_file_name] = hbl_filter(predictions[split_file_name])
                shared_invoice[split_file_name] = predictions[split_file_name]['hbl_number']

                if carrier == 1:
                    predictions[split_file_name]['table'] = table_row_filter(predictions[split_file_name]['table'])
                else:
                    predictions[split_file_name]['table'] = gen_table_filter(predictions[split_file_name]['table'])

                predictions[split_file_name]['table'] = table_remove_null(predictions[split_file_name]['table'])

            elif pred == class_indices['mbl']:
                output = PdfWriter()
                output.add_page(inputpdf.pages[page]) #pages begin at zero in pdffilewriter
                page_num = page+1 #for counters to begin at 1
                split_file_name = file_name(filename) +"_pg"+str(page_num)+".pdf"
                split_file_path = data_folder+"SPLIT/"+split_file_name
                with open(split_file_path, "wb") as outputStream:
                    output.write(outputStream) #this can be moved to only save when the split file is AP
                fd = open(split_file_path, "rb")
                carrier = classify_mbl_carrier(image)
                #print(mbl_carriers_match[carrier])
                predictions[split_file_name] = form_recognizer_one(document=fd.read(), file_name=filename, page_num=page_num, model_id=mbl_carriers_match[carrier], carrier=mbl_carriers[carrier]) 
                
                if carrier == 6: #attached cosco
                    predictions[split_file_name]['mbl_number'] = mbl_num_filter(predictions[split_file_name]['mbl_number'])
                elif carrier == 8: #evergreen
                    predictions[split_file_name]['table'] = evergreen_table_filter(predictions[split_file_name]['table'])
                elif carrier == 10: #goldstar
                    predictions[split_file_name]['table'] = goldstar_table_filter(predictions[split_file_name]['table'])
                elif carrier == 28: #tslines
                    predictions[split_file_name]['table'] = tslines_table_filter(predictions[split_file_name]['table'])
                else:
                    predictions[split_file_name]['table'] = table_filter(predictions[split_file_name]['table'])
                                
                predictions[split_file_name] = mbl_filter(predictions[split_file_name])
                predictions[split_file_name]['table'] = table_remove_null(predictions[split_file_name]['table'])
                shared_invoice[split_file_name] = predictions[split_file_name]['mbl_number']
                print(predictions)

        predictions = multipage_combine(predictions, shared_invoice)

    elif ext in ["jpg", "jpeg", "png",'.bmp','.tiff']:
        pil_image = Image.open(io.BytesIO(file_bytes)).convert('L').convert('RGB') 
        if hbl_page(pil_image) == 1:
            predictions[filename] = form_recognizer_one(document=file_bytes, file_name=filename, page_num=1, model_id="hbl_portever1")
            predictions[filename]['doc_type'] = "HBL"
    else:
        return "File type not allowed."

    for file in predictions:
        predictions = add_webservice_user(predictions, file, user_query)
        predictions[file]['process_id'] = process_id
        if predictions[file]['doc_type'] == "MBL":
            payload["MBL"] = predictions[file]#json.dumps(predictions[file])
        else:
            predictions[file]['release_type'] = find_release_type(predictions[file]['surrendered'],predictions[file]['telex_release'], predictions[file]['ebl'], predictions[file]['number_original'])
            hbl_list.append(predictions[file])

    if hbl_list: 
        payload["HBL"] = hbl_list

    #push_parsed_inv(json.dumps(payload), process_id, user_id)
    r = requests.post("https://cargomation.com:5201/redis/apinvoice/shipmentreg_hblmbl", auth=('admin', r'u\}M[6zzAU@w8YLx'), headers={'Content-Type': 'application/json'}, data=json.dumps({"user_id": user_id, "process_id: process_id"}))

    return payload
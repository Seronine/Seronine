#required packages
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import spacy
nlp=spacy.load('en_core_web_sm')
from spacy.matcher import Matcher
#code for image processing 

def pre_process(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(grey, (5, 5), 1)

    edge = cv2.Canny(blur, 135, 200)

    kernal = np.ones((5, 5))

    dilate = cv2.dilate(edge, kernal, iterations=2)

    threshold = cv2.erode(dilate, kernal, iterations=1)

    return threshold

def get_contours(img):
    biggest = np.array([])
    max_area = 0
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    points = cv2.drawContours(img_contour, biggest, -1, (255, 0, 0), 20)
    return biggest

def reshape(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), np.int32)

    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]

    return new_points
def fix_image(img, contour):
    contour = reshape(contour)

    pts1 = np.float32(contour)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # get the warp perspective transform
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # apply to the image
    final_img = cv2.warpPerspective(img, matrix, (width, height))

    #crop the image
    cropped = final_img[20:final_img.shape[0] - 20, 20:final_img.shape[1] - 20]

    return cropped

#importing file : the image
file_name = 'content/Brilliant AC.jpg'

img = cv2.imread(file_name)
#print(img.shape)
img_contour = img.copy()


#Let apply the image funtions to pre processing the image

scale_percent = 100
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

#Let apply OCR to extract the text

threshold = pre_process(img)
get_contour = get_contours(threshold)
fixed_image = fix_image(img, get_contour)

# extract text from image using tesseract

text = pytesseract.image_to_string(img)
#print('text detected: \n' + text)


#Once the text is got, now let extract the information
doc=nlp(text)
text = pytesseract.image_to_string(img)

matcher = Matcher(nlp.vocab)
matcher2 = Matcher(nlp.vocab)

#name of the company 

pattern1_A = [{"POS" :'PROPN', "OP":"+"}, {"IS_PUNCT":True}, {"ORTH":"\n"}]
pattern1_b = [{"POS" :'PROPN', "OP":"+"},  {"ORTH":"\n"}] #if the dot is forgotten
pattern1_c = [{"POS" :'PROPN', "OP":"+"}, {"IS_PUNCT":True}] #if only the dot
matcher.add("Company",[pattern1_A ,pattern1_b , pattern1_c], greedy='LONGEST')#FIRST or LONGEST : to get the name of the company

months=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec',
       'january',	'february',	'march',	'april',	'may',	'june',	'july',	'august',
        'september',	'october',	'november',	'december']
#print(months)
#Date_AA=[{'IS_DIGIT': True},  {"LOWER":{"IN" : months}},  {'IS_DIGIT': True}]
#Date 2
Date_AA=[{"SHAPE": "dd"},  {"LOWER":{"IN" : months}},  {"SHAPE": "dddd"}, {"IS_PUNCT":True}, {"ORTH":"\n"}]
Date_Ab=[{"SHAPE": "d"},  {"LOWER":{"IN" : months}},  {"SHAPE": "dddd"}, {"IS_PUNCT":True}, {"ORTH":"\n"}]
Date_Ac=[{"SHAPE": "dd"},  {"LOWER":{"IN" : months}},  {"SHAPE": "dd"}, {"IS_PUNCT":True}, {"ORTH":"\n"}]
Date_Ad=[{"SHAPE": "d"},  {"LOWER":{"IN" : months}},  {"SHAPE": "dd"}, {"IS_PUNCT":True}, {"ORTH":"\n"}] 

Date_B=[{"SHAPE": "d/d/dddd"}, {"IS_PUNCT":True}, {"ORTH":"\n"}]
Date_C=[{"SHAPE": "dd/d/dddd"}, {"IS_PUNCT":True}, {"ORTH":"\n"}]
Date_D=[{"SHAPE": "d/dd/dddd"}, {"IS_PUNCT":True}, {"ORTH":"\n"}]
Date_E=[{"SHAPE": "dd/dd/dddd"}, {"IS_PUNCT":True}, {"ORTH":"\n"}]
Date_F=[{"SHAPE": "d/d/dd"}, {"IS_PUNCT":True}, {"ORTH":"\n"}]
Date_G=[{"SHAPE": "dd/d/dd"}, {"IS_PUNCT":True}, {"ORTH":"\n"}]
Date_H=[{"SHAPE": "d/dd/dd"}, {"IS_PUNCT":True}, {"ORTH":"\n"}]
Date_I=[{"SHAPE": "dd/dd/dd"}, {"IS_PUNCT":True}, {"ORTH":"\n"}]

Date_AA2=[{"SHAPE": "dd"},  {"LOWER":{"IN" : months}},  {"SHAPE": "dddd"},  {"ORTH":"\n"}]
Date_Ab2=[{"SHAPE": "d"},  {"LOWER":{"IN" : months}},  {"SHAPE": "dddd"},  {"ORTH":"\n"}]
Date_Ac2=[{"SHAPE": "dd"},  {"LOWER":{"IN" : months}},  {"SHAPE": "dd"},  {"ORTH":"\n"}]
Date_Ad2=[{"SHAPE": "d"},  {"LOWER":{"IN" : months}},  {"SHAPE": "dd"},  {"ORTH":"\n"}] 

Date_B2=[{"SHAPE": "d/d/dddd"},  {"ORTH":"\n"}]
Date_C2=[{"SHAPE": "dd/d/dddd"},  {"ORTH":"\n"}]
Date_D2=[{"SHAPE": "d/dd/dddd"},  {"ORTH":"\n"}]
Date_E2=[{"SHAPE": "dd/dd/dddd"},  {"ORTH":"\n"}]
Date_F2=[{"SHAPE": "d/d/dd"},  {"ORTH":"\n"}]
Date_G2=[{"SHAPE": "dd/d/dd"},  {"ORTH":"\n"}]
Date_H2=[{"SHAPE": "d/dd/dd"},  {"ORTH":"\n"}]
Date_I2=[{"SHAPE": "dd/dd/dd"},  {"ORTH":"\n"}]

matcher.add("Jour_op",[Date_AA,Date_Ab,Date_Ac,Date_Ad, Date_B, Date_C, Date_D, Date_E , Date_F, Date_G, Date_H, Date_I,
                    Date_AA2,Date_Ab2,Date_Ac2,Date_Ad2, Date_B2, Date_C2, Date_D2, Date_E2 , Date_F2, Date_G2, Date_H2, Date_I2]
            , greedy='LONGEST') # Date of act




#Period of the intervention 
pattern5_A = [{"ORTH" :"between"},{"IS_ASCII":True, "OP":'+'},{"ORTH":"and"},{"POS":"NOUN"}]
pattern5_B = [{"ORTH" :"between"},{"IS_ASCII":True, "OP":'+'},{"ORTH":"and"},{"SHAPE": "dddd/dd/dd"}]

matcher.add("Period",[pattern5_A ,pattern5_B]
            , greedy='LONGEST') # Contact of the person


#person to contact 

Pattern6=[{"LOWER": "contact"},  {"POS":"PROPN", 'OP':'+'}]

matcher.add("Contact_person",[Pattern6]
            , greedy='LONGEST') # Contact of the person

#Phone of the personn 
Pattern7=[{"SHAPE": "ddd"}, {"SHAPE": "ddd"},{"SHAPE": "dddd"}]

matcher.add("Phone_num",[Pattern7], greedy='LONGEST') # Contact of the person

#Email 
Pattern8=[{"LIKE_EMAIL":True}]

matcher.add("Email_person",[Pattern8], greedy='LONGEST') # Contact of the person

# duration
names=['month','week','day','year','months','weeks','days','years',]
Pattern9=[{"LOWER":"guaranteed"},{"POS" : "ADP"},{"IS_DIGIT": True},  {"LOWER":{"IN" : names}}]

matcher.add("Duration",[Pattern9], greedy='LONGEST') # Contact of the person


doc=nlp(text)
matches=matcher(doc)


matches.sort(key=lambda x:x[1])
#print(matches)
nb=len(matches)

k=0
for i in range(0, len(matches)):
    index = matches[i]
    #tag = nlp.vocab[matches[i][0]].text
    #text = doc[matches[i][1]]
    #print(f"{index}  {tag}  {text}")
    #if index=="Company":
    if nlp.vocab[index[0]].text=="Company" and k==0:
        company_name=doc[index[1]:index[2]-1]
        k=1
    if nlp.vocab[index[0]].text=="Jour_op" :
        document_date=doc[index[1]:index[2]-1]
        
for ent in doc.ents:
    if ent.label_=='GPE':
        location=ent.text

        

for i in range(0, len(matches)):
    index = matches[i]
    if nlp.vocab[index[0]].text=="Contact_person" :
        contact_person=doc[index[1] +1:index[2]]
    if nlp.vocab[index[0]].text=="Phone_num" :
        contact_email=doc[index[1]:index[2]]
    if nlp.vocab[index[0]].text=="Email_person" :
        contact_number=doc[index[1]:index[2]]
    if nlp.vocab[index[0]].text=="Duration" :
        guaranteed=doc[index[1]+2:index[2]]
    if nlp.vocab[index[0]].text=="Period" :
        dates_between=doc[index[1]:index[2]]
        
#I removed the punctuation in case of need
b=len(company_name)
c=len(document_date)
#print(b)
if company_name[b-1].pos_ =="PUNCT":
    company_name=company_name[:-1]
if document_date[c-1].pos_ =="PUNCT":
    document_date=document_date[:-1]

print("!!!!!!!!!!!!!!!!!!!!!!    FINAL OUTPUT     -- from NGNIETCHA MERLINE !!!!!!!!!!!!!!!!!! \n")
print(f"company_name: {company_name}")
print(f"document_date: {document_date}")
print(f"location: {location}")
print(f"dates_between: {dates_between}")
print(f"contact_person: {contact_person}")
print(f"contact_email: {contact_email}")
print(f"contact_number: {contact_number}")
print(f"guaranteed: {guaranteed}")

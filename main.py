from requests.exceptions import MissingSchema
import streamlit as st
from torchvision import transforms
from model import Model
import requests
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
import torch
from googleDriveFileDownloader import googleDriveFileDownloader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-download", "--download", help="Whether you wish to download the model or not", type=bool, default=True)
args = parser.parse_args()

st.title("Example")
st.write("""
# Upload your image!
## Model can recognize images from classes stated below: \n
### |Airplane|Automobile|Bird|Cat|Deer|Dog|Frog|Horse|Ship|Truck|
""")
wrong = Image.open("wrong.jpg")
blank = Image.open("blank.png")
classes = ("airplane", "automobile" , "bird", "cat", "deer", "dog", "frog" ,"horse" ,"ship", "truck") 
transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.2225))
            ])

@st.cache
def download_model():
    gdd = googleDriveFileDownloader()
    gdd.downloadFile("https://drive.google.com/file/d/1vmPuSOOTfthkIVoZipz9p4l1J_8RGQKq/view?usp=sharing")
if args.download == True:
    download_model()

@st.cache
def load_model():
    model = Model.load_from_checkpoint(checkpoint_path = "resnet.ckpt")
    model.eval()
    return model

classifier = load_model()

def load_img_from_dir():
    uploaded_image = st.file_uploader("Choose a file from your directory")
    if uploaded_image:
        uploaded_image = Image.open(uploaded_image)
    else:
        uploaded_image = blank
    return uploaded_image

def get_response(url):
    response = requests.get(url, stream = True)
    return response    

def load_img_from_url(url):
    try:
        response = requests.get(url, stream = True)
        uploaded_image = Image.open(response.raw)
    except MissingSchema:
        uploaded_image = blank
    finally:
        return uploaded_image

def get_predictions(model_out):
    return F.softmax(model_out, dim=1)

def predict(model, image):
    image = transform(image)
    image = image.unsqueeze(0)
    output = model(image)
    predictions = F.softmax(output, dim=1)
    _, pred = torch.max(output, 1)
    return pred, predictions

upload_type = st.radio(
                    "Select the way you want the image to be uploaded", 
                    ("Upload from directory", "URL"))

def load_image(upload_type):
    if(upload_type == "Upload from directory"):
        img = load_img_from_dir()
    else:
        url = st.text_input("Enter image URL", "https://www.naturschaetze-suedwestfalens.de/var/sauerland/storage/images/media/bilder/naturschaetze/buchfotos/05_p3_laubfrosch-mbd/507661-1-ger-DE/05_P3_Laubfrosch-MBD_front_large.jpg")
        img = load_img_from_url(url)
    return img

def wrap():
    try:
        img = load_image(upload_type)
        st.image(img)
        if img != blank:
            prediction, prob = predict(classifier, img)
            st.write("Predicted label: ", classes[prediction.item()])
            st.write(f"Model is {(torch.max(prob)*100):.3f} % sure that the label is correct")
    except (UnboundLocalError, UnidentifiedImageError):
        st.write("There was a problem loading the image, please try again")
        st.image(wrong)


wrap()


from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from torch import nn, optim
import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
from torchvision import transforms

app = FastAPI()


labels = [
    "Bakteri Red. Gunakan antibiotik seperti oxytetracycline, isolasi ikan terinfeksi, perbaiki kualitas air dengan pergantian rutin dan aerasi, serta disinfeksi kolam dengan larutan klorin atau formalin.",
    "Aeromoniasis. Pengobatan dengan antibiotik seperti oxytetracycline, jaga kualitas air optimal, tambahkan probiotik ke dalam pakan, dan isolasi ikan yang terinfeksi untuk perawatan intensif.",
    "Bacteri grill. Gunakan antibiotik seperti chloramphenicol, tingkatkan aerasi untuk pasokan oksigen, jaga kualitas air, dan tambahkan garam ikan (NaCl) untuk mengurangi infeksi.",
    "Jamur. Obati dengan antijamur seperti malachite green atau formalin, isolasi ikan terinfeksi, tambahkan garam ikan (NaCl), dan pastikan air bersih dengan sirkulasi baik.",
    "Sehat. tetap waspada",
    "Parasit. Gunakan obat parasit seperti praziquantel atau metronidazole, isolasi ikan yang terinfeksi, jaga kualitas air, dan bersihkan kolam serta ganti air secara berkala.",
    "Ekor putih. Pengobatan dengan antibiotik seperti tetracycline, isolasi ikan yang terinfeksi, pastikan air bersih dengan pH seimbang dan oksigen cukup, serta tambahkan garam ikan (NaCl) untuk membantu penyembuhan."
]


model = models.mobilenet_v3_large(pretrained=True)
output_size = 7

for param in model.parameters():
  param.requires_grad = False

class Fishmodel (nn.Module) :
  def freeze (self) :
    for param in self.model.parameters():
      param.requires_grad = False
  def unfreeze (self) :
    for param in model.parameters():
      param.requires_grad = True
  def __init__(self, output_size) :
    super ().__init__()
    self.model = models.mobilenet_v3_large(pretrained=True)
    self.freeze()
    self.model.classifier = nn.Sequential(
    nn.Linear ( 960, output_size),
    nn.Softmax()
)

  def forward (self,x) :
    return self.model(x)

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Load the model
model = Fishmodel(output_size)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()
transform = get_transform()
def predict_image(img):
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        predicted_class_index = predicted.item()
        predicted_class = labels[predicted_class_index]

    return f"Ikan anda mengalami penyakit {predicted_class}."

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))

    result = predict_image(img)

    return {"prediction": result}

if __name__ == "__main__":
    import nest_asyncio
    import uvicorn
    from pyngrok import ngrok

    # Apply the nest_asyncio fix
    nest_asyncio.apply()

    # Authenticate ngrok with your authtoken
    authtoken = "2hJPeYO2aRVZlLfv5YoOP26anBS_722MWnwYo8hZ2nyLe9spe"
    ngrok.set_auth_token(authtoken)

    # Create a tunnel to the web server
    public_url = ngrok.connect(8000, bind_tls=True)
    print("Public URL:", public_url)

    # Run the FastAPI app
    uvicorn.run(app, host='0.0.0.0', port=8000)
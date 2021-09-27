from fastapi import FastAPI 
from fastapi import UploadFile, File
import transcrib.py
import uvicorn

app=FastAPI()


@app.get('/index')
def msg():
    return "OCR Web App"

@app.post('/api/transcribe')

def img_transcrib(file: UploadFile= File(...)):
    #read image
    image= read_img(await, file)
    

if __name__== "__main__":
    uvicorn.run(app, port=8080, host='127.0.0.1')
    
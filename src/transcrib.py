from PIL import Image
from io import BytesIO

def read_img(img_encode):
    pil_img=Image.open(BytesIO(img_encode))
    
    return pil_img

def preprocess(image:Image.image):
    input_shape={224,224}
    
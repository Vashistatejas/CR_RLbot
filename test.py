
import mss
import numpy as np
from PIL import Image
import os
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import json
import base64
import io

load_dotenv()



# Capture full screen
def capture_screen(monitor_id=1):
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_id]
        img = sct.grab(monitor)

        frame = np.array(img)

        #  FIX: BGRA → RGB
        frame = frame[:, :, :3]          # drop alpha
        frame = frame[:, :, ::-1]        # BGR → RGB

        return Image.fromarray(frame, mode="RGB")

# Crop to Clash Royale region (tune once)
CLASH_BOX = (630, 50, 1220, 1030)  # (left, top, right, bottom)

def crop_clash(img):
    return img.crop(CLASH_BOX)


def preprocess_image(img):
    # Resize to stay safe
    img = img.resize((1280, 720))
    return img

def pil_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")



CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)
        
#VISION_PROMPT = load_prompt("vision_prompt.txt")    

def get_game_state_from_screen():
    # 1. Screenshot
    img = capture_screen()

    # 2. Crop
    img = crop_clash(img)

    # 3. Resize
    #img = preprocess_image(img)

    img = img.convert("RGB")

    



# run inference on a local image
    print(CLIENT.infer(
        img, 
        model_id="clash-royale-cylln-8joce/7"
    ))

if __name__ == "__main__":
    state = get_game_state_from_screen()
    if state:
        print(json.dumps(state, indent=2))

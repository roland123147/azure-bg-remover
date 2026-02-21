import cv2
import numpy as np
from flask import Flask, request, send_file, render_template_string
import io

app = Flask(__name__)

HTML = '''
<!doctype html>
<title>Azure Background Remover</title>
<h1>Upload Image to Remove Background</h1>
<form method=post enctype=multipart/form-data action="/remove-bg">
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    file = request.files['file']
    
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    
    h, w = img.shape[:2]
    rect = (int(w*0.05), int(h*0.05), int(w*0.9), int(h*0.9))
    
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    
    is_success, buffer = cv2.imencode(".png", img)
    return send_file(io.BytesIO(buffer), mimetype='image/png')

if __name__ == '__main__':
    app.run()

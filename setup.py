import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name='buffalo_l',
    allowed_modules=['detection', 'recognition', 'landmark'],
    providers=['CPUExecutionProvider']  
)
app.prepare(ctx_id=0, det_size=(128, 128))

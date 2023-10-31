from fastapi import APIRouter, File, UploadFile

from api.utils.readers import read_image_file
from api.utils.classificator import predict

image_classification_router = APIRouter(
    prefix='/image', tags=['image_classification']
)


@image_classification_router.post('/classification', status_code=200)
async def image_classification(file: UploadFile = File(...)):
    try:
        image = read_image_file(await file.read())
        prediction = predict(image)
        return prediction
    except Exception as e:
        return {'error': str(e)}

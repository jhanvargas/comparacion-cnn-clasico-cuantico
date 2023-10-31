# External libraries
import os
import uvicorn

from fastapi import FastAPI

app = FastAPI(title='CNN', version='0.1.0')

for root, dirs, files in os.walk(os.path.join('api', 'routers')):
    if '__' not in root:
        path = root.split(os.sep)
        for file in files:
            if '__' not in file:
                file, _ = os.path.splitext(file)
                path_import = f'{".".join(path)}.{file}'
                module = __import__(path_import, globals(), locals(), [f'{file}_router'])
                router = getattr(module, f'{file}_router')
                app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(app)

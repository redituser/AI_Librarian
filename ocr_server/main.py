import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from paddleocr import PaddleOCR
from PIL import Image

app = FastAPI( #어플리케이션 생성
    title = "PaddleOCR FastAPI Server",
    description = "이미지에서 텍스트를 추출하는 OCR 입니다"
)


try :
    ocr = PaddleOCR(use_angle_cls=False , lang='korean' , use_gpu=False)
    print("Paddle 모델 로딩 완료")
except Exception as e :
    print(f"모델 로딩중 에러발생:{e}")
    ocr = None


#엔드포인트 정의

@app.post("/ocr", summary="이미지에서 텍스트 추출" , description="업로드된 이미지 파일 텍스트와 신뢰도 점수를 추출")
async def perform_ocr(file : UploadFile = File(...)):
    #여기서 이미지 파일 받고 ocr 후 결과를 json 으로 반환

    if not ocr:
        raise HTTPException(status_code=500 , detail="ocr 모델이 로드되지 않음")

    # -- 이미지 파일 처리
    contents = await file.read()

    try:
        pil_image = Image.open(io.BytesIO(contents))#이미지를 바이트 코드로 ?
    except Exception:
        raise HTTPException(status_code=400 , detail = "유효하지 않은 이미지 파일입니다") #raise 가 뭐였더라


    #paddle ocr 이 요구하는 방식으로 이미지 변환 numpy 형식을 요구함
    image_np = np.array(pil_image)

    if len(image_np.shape) == 2: #?? 왜 2와 비교하는지?
        image_np = cv2.cvtColor(image_np , cv2.COLOR_GRAY2BGR)
    else:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    
    # ocr 실행
    try:
        result = ocr.ocr(image_np , cls = False) # ocr.ocr? 실행 함수인가
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ocr처리중 에러{str(e)}")
        
    #결과가 잘나오면 json으로 포매팅
    if not result or not result[0]: #result 자체 값이 없거나 , 첫번째 인수의 값이 없다면  ?
        return {"results" : []} #이건 무슨 구조인가?
        

    formatted_results =[]
    for line in result[0]: #result의 구성이 어떻길래?
        text = line[1][0]
        confidence = line[1][1]
        bounding_box = line[0]

        formatted_results.append({
            "text":text,
            "confidence" : float(f"{confidence:.4f}"),
            "bounding_box" : bounding_box
        })


    return {"results" : formatted_results} #{"results" : [a , b , c] } 이런구조인가


@app.get("/", summary="서버 상태 확인")
def read_root():
    return {"status": "paddle ocr api 작동중"}
    


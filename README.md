# pose-transformer-with-real-transoformer
이미지를 넣어서 pose에 따라 이미지를 변환시키는 건데 만들긴 한거 같은데 학습 자료가 별로 안 좋아서... 잘 안되는 거 같습니다. 일단 모델은 올려봅니다.
수정할 부분이나 그런거 있으면 좀 알려주세요.

제가 직접 만든 모델을 한번 구경해보고 싶어서요. 지금은 좀 엉망으로 나오는지라....

제 이메일등(ryujanghyun00@icloud.com)으로 자료 보내 주시면 학습 한번 해볼게요. 이미지 사이즈는 WxH가 144x256로 만들었습니댜.

사용 방법은 

1. python -m venv ./ 이후에 source bin/activate 작동.
2. pip install -r requirements.txt
3. pth.zip과 outdate.zip 그리고 fashion.zip을 압축 풀어 넣기. (학습시 fashion.zip or outdate.zip필요 실행만 해볼 요량이면 pth.zip만 압축 풀기)  ( 다운로드 경로 https://drive.google.com/drive/folders/1jNeGA9nbLsa37PXgRcy-lhROPh58zOnp?usp=drive_link  )
  (학습 순서
    1. fashion.zip파일을 푼 이후에
       python change_npnew.py
       그러면 outdate폴더에 npy파일이 생성됨.
    2. 이후
       python mytrain.py
       이러면 학습이 진행됨
       pth폴더에 torch모델이 저장됨.
  ) 
5. $ python mymain.py 실행
   (테스트시에 컴퓨터에 웹카메라등을 설치가 필요합니다. 웹 카메라는 90도 반시계 방향으로 돌린 이후 사용합니다.)
   (pth폴더의 파일의 이름을 확인한 이후 mymain.py에 파일 이름을 교체합니다.)

참조 논문은
https://github.com/prasunroy/pose-transfer?tab=readme-ov-file 이걸 참조해서 사이에 transformer를 넣어 작성해 봤습니다. 더 잘되는 것 같긴 합니다.



2026/2/21
gan 모델을 추가하였습니다. 학습속도가 비약적으로 빨라 졌습니다.
그리고 loss 모델을 수정했습니다.

# 필요한 라이브러리 임포트
from math import sqrt, ceil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os

def plot_ckalist_resume(cka_list, save_name):
    # CKA 리스트의 길이 계산
    n = len(cka_list)
    
    # 서브플롯 행과 열 수 계산
    y = ceil(sqrt(n))
    if n == sqrt(n)*sqrt(n):
        x = y
    elif (y-1) * y < n:
        x = y
    else:
        x = y - 1
    print("x | y :", x, y)    
    
    # 전체 그림 생성
    fig = plt.figure(figsize=(y*4, x*4), frameon=False)

    sc = None
    for i, cka in enumerate(cka_list):
        # 서브플롯 추가
        ax = fig.add_subplot(x, y, i+1)
        ll = cka.shape[0]
        
        # CKA 행렬 이미지로 표시
        sc = ax.imshow(cka, cmap='magma', vmin=0.0, vmax=1.0)
        
        # x축 틱 설정
        step = max(1, int(ll/5))

        tick = [i for i in range(0, ll, step)]
        ax.set_xticks(tick) 
        
        # y축 틱 제거
        ax.set_yticks([]) 
        
        # y축 반전
        ax.axes.invert_yaxis()
    
    # 컬러바 위치 및 크기 설정
    l, b, w, h = 0.92, 0.35, 0.015, 0.35
    rect = [l, b, w, h] 
    cbar_ax = fig.add_axes(rect) 

    # 컬러바 추가
    plt.colorbar(sc, cax=cbar_ax)

    # 그림 저장
    plt.savefig(f'{save_name}.png', dpi=700)   

# 주의: 이 함수는 PyTorch 텐서를 직접 다루지 않습니다.
# CKA 계산 결과가 NumPy 배열 형태로 제공된다고 가정합니다.
# 만약 PyTorch 텐서를 직접 다룰 경우, 아래와 같이 수정이 필요할 수 있습니다:
# cka = cka.cpu().numpy() if isinstance(cka, torch.Tensor) else cka
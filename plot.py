# 필요한 라이브러리 임포트
from math import sqrt, ceil
import matplotlib.pyplot as plt
# import seaborn as sns
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

# 필요한 라이브러리 임포트
import torch
import argparse
import os
import pickle
import numpy as np
from plot import *

def plot_cka_map(cka_file_name, plot_name, base_dir):
    
    


    # GPU 설정

    #base_dir + cka_file_name폴더가 없으면 폴더를 만든다.
    if not os.path.exists(os.path.join(base_dir, cka_file_name)):
        os.makedirs(os.path.join(base_dir, cka_file_name))
    # CKA 결과 파일 경로 설정
    cka_dir = os.path.join(base_dir, cka_file_name, cka_file_name + "_heatmap.pkl")


    # CKA 결과 불러오기
    with open(cka_dir, 'rb') as f:
        cka = pickle.load(f)
    
    qkv_activations = [(i * 4) + 1 for i in range(0, 12)]
    proj_activations = [(i * 4) + 2 for i in range(0, 12)]
    mlp_fc1_activations = [(i * 4) + 3 for i in range(0, 12)]
    mlp_fc2_activations = [(i * 4) + 4 for i in range(0, 12)]
        
    # 전체 레이어에 대한 CKA 결과 플롯 생성
    plot_dir = os.path.join(base_dir, plot_name)
    qkv_plot_dir = os.path.join(plot_dir,  'cka_qkv')
    proj_plot_dir = os.path.join(plot_dir,  'cka_proj')
    mlp_fc1_plot_dir = os.path.join(plot_dir,  'cka_mlp_fc1')
    mlp_fc2_plot_dir = os.path.join(plot_dir,  'cka_mlp_fc2')
    
    #qkv
    qkv_cka1 = cka[qkv_activations]
    qkv_cka1 = qkv_cka1[:,qkv_activations]
    print(cka.shape, qkv_cka1.shape)
    #pickle로 저장한다.
    with open(os.path.join(plot_dir, 'cka_qkv.pkl'), 'wb') as f:
        pickle.dump(qkv_cka1, f)
    #proj
    proj_cka1 = cka[proj_activations]
    proj_cka1 = proj_cka1[:,proj_activations]
    print(cka.shape, proj_cka1.shape)
    with open(os.path.join(plot_dir, 'cka_proj.pkl'), 'wb') as f:
        pickle.dump(proj_cka1, f)
    
    #mlp_fc1
    mlp_fc1_cka1 = cka[mlp_fc1_activations]
    mlp_fc1_cka1 = mlp_fc1_cka1[:,mlp_fc1_activations]
    print(cka.shape, mlp_fc1_cka1.shape)
    with open(os.path.join(plot_dir, 'cka_mlp_fc1.pkl'), 'wb') as f:
        pickle.dump(mlp_fc1_cka1, f)
    
    #mlp_fc2
    mlp_fc2_cka1 = cka[mlp_fc2_activations]
    mlp_fc2_cka1 = mlp_fc2_cka1[:,mlp_fc2_activations]
    
    print(cka.shape, mlp_fc2_cka1.shape)
    with open(os.path.join(plot_dir, 'cka_mlp_fc2.pkl'), 'wb') as f:
        pickle.dump(mlp_fc2_cka1, f)
    

    
    
    
    plot_ckalist_resume([cka], plot_dir)
    plot_ckalist_resume([qkv_cka1], qkv_plot_dir)
    plot_ckalist_resume([proj_cka1], proj_plot_dir)
    plot_ckalist_resume([mlp_fc1_cka1], mlp_fc1_plot_dir)
    plot_ckalist_resume([mlp_fc2_cka1], mlp_fc2_plot_dir)
# plot_cka_map('cka_not_quantized_result.pkl', 'cka_not_quantized_result.png')
import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_and_plot_diagonal(pickle_file):
    # pickle 파일 불러오기
    with open(f'{pickle_file}.pkl', 'rb') as f:
        heatmap = pickle.load(f)

    
    # 대각 성분 추출
    diagonal = np.diag(heatmap)

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(diagonal, marker='o')
    plt.title('Diagonal Elements of CKA Matrix')
    plt.xlabel('Layer Index')
    plt.ylabel('CKA Value')
    plt.ylim(0, 1)  # CKA 값의 범위는 0에서 1 사이입니다
    plt.grid(True)

    # 그래프 저장
    plt.savefig(f'{pickle_file}_diagonal.png', dpi=300, bbox_inches='tight')
    plt.close()

    return diagonal


import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_diagonal(pickle_file):
    with open(f'{pickle_file}.pkl', 'rb') as f:
        heatmap = pickle.load(f)
    return np.diag(heatmap)

def plot_all_diagonals(pickle_files, labels):
    plt.figure(figsize=(15, 8))

    qkv_activations = [i * 4 + 1 for i in range(12)]
    proj_activations = [i * 4 + 2 for i in range(12)]
    mlp_fc1_activations = [i * 4 + 3 for i in range(12)]
    mlp_fc2_activations = [i * 4 + 4 for i in range(12)]

    all_activations = qkv_activations + proj_activations + mlp_fc1_activations + mlp_fc2_activations
    max_activation = max(all_activations)

    for pickle_file, label in zip(pickle_files, labels):
        diagonal = load_diagonal(pickle_file)
        
        if label == 'Comprehensive':
            plt.plot(range(len(diagonal)), diagonal, marker='o', label=label)
        elif label == 'QKV':
            values = [diagonal[i] if i < len(diagonal) else None for i in range(12)]
            plt.plot(qkv_activations, values, marker='o', label=label)
        elif label == 'Proj':
            values = [diagonal[i] if i < len(diagonal) else None for i in range(12)]
            plt.plot(proj_activations, values, marker='o', label=label)
        elif label == 'MLP FC1':
            values = [diagonal[i] if i < len(diagonal) else None for i in range(12)]
            plt.plot(mlp_fc1_activations, values, marker='o', label=label)
        elif label == 'MLP FC2':
            values = [diagonal[i] if i < len(diagonal) else None for i in range(12)]
            plt.plot(mlp_fc2_activations, values, marker='o', label=label)

    plt.title('Diagonal Elements of CKA Matrices')
    plt.xlabel('Layer Index')
    plt.ylabel('CKA Value')
    plt.ylim(0, 1)
    plt.xlim(0, max_activation + 1)
    plt.xticks(range(0, max_activation + 1, 4))
    plt.grid(True)
    plt.legend()

    plt.savefig('all_diagonals_matched.png', dpi=300, bbox_inches='tight')
    plt.show()
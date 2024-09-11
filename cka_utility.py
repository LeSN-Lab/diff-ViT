from dataset_utility import *
from models import *
import pickle

def normalize_activations(act):
    # 입력 텐서를 2D로 재구성합니다. 첫 번째 차원은 유지하고 나머지는 평탄화합니다.
    act = act.view(act.size(0), -1)

    # 각 샘플(행)에 대해 L2 norm을 계산합니다.
    act_norm = torch.norm(act, p=2, dim=1, keepdim=True)

    # 0으로 나누는 것을 방지하기 위해 작은 값을 더합니다.
    act_norm = act_norm + 1e-8

    # 각 샘플을 해당 norm으로 나누어 정규화합니다.
    act = act / act_norm

    return act
#torch model의 layers의 수를 확인한다.
from efficient_CKA import *

def get_activations(images, model, bit_config, device, normalize_act=False):
    model = model.to(device)

    



    def get_module_path(module):
        return f"{module.__class__.__module__}.{module.__class__.__name__}"

    activations = []
    layer_info = []
    
    from models.vit_fquant import Attention, Mlp
    def hook_return(index):
        def hook(module, input, output):
            if isinstance(module, Attention):
                activations.append(module.qkv_output)
                layer_info.append({
                'relative_index': len(activations) - 1,
                'absolute_index': index,
                'name': module.__class__.__name__,
                'layer_type': type(module),
                'path': get_module_path(module)

                })
                
            elif isinstance(module, Mlp):
                activations.append(module.fc1_output)
                layer_info.append({
                'relative_index': len(activations) - 1,
                'absolute_index': index,
                'name': module.__class__.__name__,
                'layer_type': type(module),
                'path': get_module_path(module)

                })
            else:
                activations.append(output)
                layer_info.append({
                    'relative_index': len(activations) - 1,
                    'absolute_index': index,
                    'name': module.__class__.__name__,
                    'layer_type': type(module),
                    'path': get_module_path(module)

                })
            

        return hook

    hooks = []

    if bit_config is None:
        for index, layer in enumerate(model.modules()):
            if type(layer) in [QConv2d, QLinear, Attention, Mlp]:
                hooks.append(layer.register_forward_hook(hook_return(index)))
    else:
        for index, layer in enumerate(model.modules()):
            if type(layer) in [QConv2d, QLinear]:
                hooks.append(layer.register_forward_hook(hook_return(index)))
    # 모델을 통해 이미지를 전달합니다.
    images = images.cuda()
    _ = model(images, bit_config = bit_config, plot=False)

    # 등록된 후크를 제거합니다.
    for h in hooks:
        h.remove()





    # layer_info와 activations를 절대 인덱스를 기준으로 정렬
    sorted_indices = sorted(range(len(layer_info)), key=lambda k: layer_info[k]['absolute_index'])
    layer_info = [layer_info[i] for i in sorted_indices]
    activations = [activations[i] for i in sorted_indices]

    # 상대 인덱스 재할당
    for i, info in enumerate(layer_info):
        info['relative_index'] = i


    if normalize_act:
        activations = [normalize_activations(act) for act in activations]
    # for info in layer_info:
    #     print(f"Layer {info['relative_index']}(absolute: {info['absolute_index']}): {info['name']} (Type: {info['layer_type']}, Path: {info['path']})")
    return activations
    # 정렬된 레이어 정보 출력
    

    import pickle
def compute_cka_internal(model, use_batch = True,
                         use_train_mode = False,
                         normalize_act = False,
                         cka_batch = 50,
                         cka_batch_iter = 10,
                         cka_iter = 10,
                         result_name = 'cka_result.pkl',
                         device = 'cuda'
                         ):
    model.eval()

    sample_cka_dataset = get_dataset(cka_batch)

    sample_cka_dataset = next(iter(sample_cka_dataset))

    sample_images, _ = sample_cka_dataset
    # n_layers = len(list(not_quantized_model.children()))
    # n_layers = len([layer for layer in model.modules() if isinstance(layer, (nn.Conv2d, nn.Linear))])

    sample_activations = get_activations(
        images = sample_images, 
        model = model, 
        bit_config=None, 
        device = device,
        normalize_act = normalize_act)
    n_layers = len(sample_activations)

    cka = MinibatchCKA(n_layers)
    

    if use_batch:
        for index in range(cka_iter):
            #cka_batch만큼, shuffle해서, 데이터셋을 가져온다.
            cka_dataset = get_dataset(cka_batch)
            current_iter = 0
            for images, _ in cka_dataset:
                model_get_activation = get_activations(images, model, None, normalize_act) #각 모델의 레이어별 활성화를 가져온다.

                cka.update_state(model_get_activation) #레이어 마다의 activation을 다 가져옴. 예를 들어 24 * 50 * feature^2. 
                
                if current_iter > cka_batch_iter:
                    break
                current_iter += 1
            print("현재 반복:", index)
    else:
        cka_dataset = get_dataset(cka_batch)
        all_images = []
        for images, _ in cka_dataset:
            all_images.append(images)
        cka.update_state(get_activations(all_images, model, None, normalize_act))
    heatmap = cka.result().cpu().numpy()
    # result_name을 폴더 이름으로 사용
    folder_name = result_name

    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 파일 이름 (예: 'h_map.pkl')
    file_name = '_heatmap.pkl'

    # 전체 파일 경로
    full_path = os.path.join(folder_name, file_name)

    # 파일 저장
    with open(full_path, 'wb') as f:
        pickle.dump(heatmap, f)
from DDV_CKA import *
def compute_cka_with_adversarial(model1, model2, use_batch = True,
                         normalize_act = False,
                         cka_batch = 50,
                         cka_batch_iter = 10,
                         cka_iter = 10,
                         result_name = 'cka_result.pkl',
                         model1_bit_config = None,
                         model2_bit_config = None,
                         args = None
                         ):
    model1.eval()
    model2.eval()

    sample_cka_dataset = get_dataset(n = cka_batch, args= args)

    sample_cka_dataset = next(iter(sample_cka_dataset))

    sample_images, sample_labels = sample_cka_dataset
    cka_attack_net1 = AttackPGD(model1, epsilon=0.06, step_size=0.01, num_steps=50, bit_config = model1_bit_config, args = args)
    
    
    
    # cka_attack_net2 = AttackPGD(model2, epsilon=0.06, step_size=0.01, num_steps=50, bit_config = bit_config)
    #@To Do: cka_attack_net2를 직접 사용해보기
    cka_attack_net2 = cka_attack_net1 #모델1과 같은 공격 네트워크를 사용한다. 
    
    
    
    

    sample_activations = get_activations(
        images = sample_images,
        model = model1,
        bit_config=model1_bit_config,
        device=args.device,
        normalize_act=normalize_act)
    n_layers = len(sample_activations)

    cka = MinibatchAdvCKA(n_layers)
    

    if use_batch:
        for index in range(cka_iter):
            #cka_batch만큼, shuffle해서, 데이터셋을 가져온다.
            cka_dataset = get_dataset(
                n = cka_batch,
                args=args)
            current_iter = 0
            for images, labels in cka_dataset:
                adv_images = cka_attack_net1.gen_adv_inputs(images, labels)
                
                model1_get_activation = get_activations(
                    images=images,
                    model = model1,
                    bit_config=model1_bit_config,
                    device=args.device,
                    normalize_act=normalize_act) #각 모델의 레이어별 활성화를 가져온다.
                model1_get_adv_activation = get_activations(
                    images=adv_images,
                    model = model1,
                    bit_config=model1_bit_config,
                    device=args.device,
                    normalize_act=normalize_act)
                
                model2_get_activation = get_activations(
                    images=images,
                    model = model2,
                    bit_config=model2_bit_config,
                    device=args.device,
                    normalize_act=normalize_act)
                model2_get_adv_activation = get_activations(
                    images=adv_images,
                    model = model2,
                    bit_config=model2_bit_config,
                    device=args.device,
                    normalize_act=normalize_act)
                
                cka.update_state(model1_activations=model1_get_activation,
                                 model1_adv_activations=model1_get_adv_activation,
                                 model2_activations=model2_get_activation,
                                 model2_adv_activations=model2_get_adv_activation) #레이어 마다의 activation을 다 가져옴. 예를 들어 24 * 50 * feature^2. 
                
                if current_iter > cka_batch_iter:
                    break
                current_iter += 1
            print("현재 반복:", index)
    else:
        cka_dataset = get_dataset(
            n = cka_batch, 
            args=args)
        all_images = []
        all_labels = []
        for images, labels in cka_dataset:
            all_images.append(images)
            all_labels.append(labels)
            all_adv_images = cka_attack_net1.gen_adv_inputs(
                inputs=all_images,
                labels=all_labels
                )
        cka.update_state(
            model1_activations=get_activations(
                images=all_images,
                model=model1,
                bit_config=model1_bit_config,
                device=args.device,
                normalize_act=normalize_act
                ),
            model1_adv_activations=get_activations(
                images=all_adv_images,
                model= model1,
                bit_config=model1_bit_config,
                device=args.device,
                normalize_act=normalize_act),
            model2_activations=get_activations(
                images=all_images,
                model = model2,
                bit_config=model2_bit_config,
                device=args.device,
                normalize_act=normalize_act),
            model2_adv_activations=get_activations(
                images=all_adv_images,
                model=model2,
                bit_config=model2_bit_config,
                device=args.device,
                normalize_act=normalize_act),
            )
    heatmap = cka.result().cpu().numpy()
    #result_name에 해당하는 폴더를 만든다.
    if not os.path.exists(result_name):
        os.makedirs(result_name)
    #result_name 폴더에 result_name_heatmap.pkl로 heatmap을 저장한다.
    with open(os.path.join(result_name, result_name + '_heatmap.pkl'), 'wb') as f:
        pickle.dump(heatmap, f)
    
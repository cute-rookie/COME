import os, torch
import numpy as np
import torch, json
import shutil, base64
from openai import OpenAI
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from diffusers import StableDiffusion3Pipeline
from jinja2.compiler import generate
from modelscope import snapshot_download
from datetime import datetime
from tqdm import tqdm
from typing import Optional, List
from torch import Tensor
from transformers import AutoModel
from tqdm import tqdm

def get_att_in_celeba(pred_info):
    json_data = json.loads(pred_info)
    gender_str = json_data['Gender']
    attractivenese_str = json_data['Attractive']
    if gender_str == 'male':
        gender_label = 1
    else:
        gender_label = 0
    if attractivenese_str == 'yes':
        attractivenese_label = 1
    else:
        attractivenese_label = 0
    return gender_label, attractivenese_label

def get_labeled_data(filename, reset, nclass, n_labeled_samples, targets, ds, ta, sa):
    path = f'./dataset/{ds}/{ds}_{ta}_{sa}_{nclass}/{filename}'

    generate_data = filename == f'train_{n_labeled_samples}.txt'
    generate_data = generate_data and (reset or not os.path.exists(path))

    if not generate_data:
        print(f'Loading {filename} from {path}...')
        data_index = torch.load(path)
    else:
        path_test = f'./dataset/{ds}/{ds}_{ta}_{sa}_{nclass}/test_{n_labeled_samples}.txt'
        path_db = f'./dataset/{ds}/{ds}_{ta}_{sa}_{nclass}/database_{n_labeled_samples}.txt'
        if os.path.exists(path):
            os.remove(path)
            os.remove(path_test)
            os.remove(path_db)
        train_data_index = []
        query_data_index = []
        db_data_index = []

        data_id = np.arange(len(targets))  # [0, 1, ...]

        for i in range(nclass):
            if n_labeled_samples % nclass != 0:
                assert False, 'n_labeled_samples must be divisible by nclass'
            sub_n_labeled_samples = n_labeled_samples // nclass
            class_mask = targets == i
            index_of_class = data_id[class_mask].copy()  # index of the class [2, 10, 656,...]
            np.random.shuffle(index_of_class)

            if nclass == 2:
                query_n = 250
            else:
                query_n = 20

            index_for_query = index_of_class[:query_n].tolist() # for test
            index_for_db = index_of_class[query_n:].tolist() # for database
            # index_for_train = index_for_db.copy() # for train
            # choose samplers for train from the db
            if sub_n_labeled_samples < len(index_for_db):
                index_for_train = np.random.choice(index_for_db, sub_n_labeled_samples, replace=False).tolist()
            else:
                index_for_train = index_for_db.copy()

            train_data_index.extend(index_for_train)
            query_data_index.extend(index_for_query)
            db_data_index.extend(index_for_db)
        if len(train_data_index) != n_labeled_samples:
            while len(train_data_index) != n_labeled_samples:
                index_for_train = np.random.choice(db_data_index, 1, replace=False).tolist()
                if index_for_train not in train_data_index:
                    train_data_index.extend(index_for_train)

        train_data_index = np.array(train_data_index)
        query_data_index = np.array(query_data_index)
        db_data_index = np.array(db_data_index)

        print('train_data_index', train_data_index.shape)
        print('query_data_index', query_data_index.shape)
        print('db_data_index', db_data_index.shape)

        os.makedirs(f'./dataset/{ds}/{ds}_{ta}_{sa}_{nclass}', exist_ok=True)
        torch.save(train_data_index, f'./dataset/{ds}/{ds}_{ta}_{sa}_{nclass}/train_{n_labeled_samples}.txt')
        torch.save(query_data_index, f'./dataset/{ds}/{ds}_{ta}_{sa}_{nclass}/test_{n_labeled_samples}.txt')
        torch.save(db_data_index, f'./dataset/{ds}/{ds}_{ta}_{sa}_{nclass}/database_{n_labeled_samples}.txt')

        data_index = {
            f'train_{n_labeled_samples}.txt': train_data_index,
            f'test_{n_labeled_samples}.txt': query_data_index,
            f'database_{n_labeled_samples}.txt': db_data_index
        }[filename]

    return data_index, generate_data

def get_relations(relation_path, sample_path, ori_train_imgs):
    if not os.path.exists(relation_path):
        return False
    else:
        relations = torch.load(relation_path)
        for i in ori_train_imgs:
            if i not in relations:
                return False
        if os.path.exists(sample_path):
            imgs = os.listdir(sample_path)
            if len(imgs) + len(ori_train_imgs) != len(relations):
                return False
            if len(imgs) == 0:
                return False
            else:
                for i in imgs:
                    if i not in relations:
                        return False
        else:
            return False
        return True

def load_CDM(config):
    # load conditional diffusion model#

    cache_dir = "./LLM/diff" #
    type = config['sample']['LLM_for_image_type']
    if type == 0:
        model_dir = snapshot_download("AI-ModelScope/stable-diffusion-3.5-medium", cache_dir=cache_dir, local_files_only=True) # Can be used in 24GB GPU
    elif type == 1:
        model_dir = snapshot_download("AI-ModelScope/stable-diffusion-3.5-large", cache_dir=cache_dir, local_files_only=True) # Can be used in 48GB GPU
    else:
        assert False, 'Invalid LLM_for_image_type'

    pipe = StableDiffusion3Pipeline.from_pretrained(model_dir, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe

def generate_samples_use_cdm(ori_img_list, ta, sa, config):
    """
    generate samples using CDM, with same target attribute and different sensitive attribute
    the sensitive attribute is always the binary classifications
    :param ori_img_list: original image list,
    :param ta: target attribute
    :param sa: sensitive attribute
    :return: none, save generated data to the generated_data folder
    """
    data_root = config['dataset_kwargs']['data_root']
    n_labeled_samples = len(ori_img_list)
    nclass = config['arch_kwargs']['nclass']
    ds = config['dataset']
    saved_path = f'./dataset/generated_data/{ds}_{ta}_{sa}_{nclass}/samples_{n_labeled_samples}'
    if os.path.exists(saved_path):
        shutil.rmtree(saved_path)
    pipe = load_CDM(config)
    for filename in tqdm(ori_img_list, desc='Generating samples'):
        if ds.startswith('utkface'):
            if ta == 'age':
                generated_age = filename.split('_')[0]
                assert sa == 'ethnicity', 'when the target attribute is age, the sensitive attribute must be ethnicity, detailed in ./configs/reasonableness_judgment function'

                gender = filename.split('_')[1]
                if gender == '0': generated_gender ='male'
                else: generated_gender = 'female'
                if filename.split('_')[2] != '0': # if the ethnicity is not White, generate a photo with White ethnicity
                    generated_ethnicity = 'White'
                    description_str = f"A photograph of a person's face that meets the following requirements, gender: {generated_gender}, age: {generated_age} years old, ethnicity: {generated_ethnicity}"
                else:
                    choose_ethnicity = ['Black', 'Asian', 'Indian', 'Hispanic', 'Middle Eastern']
                    generated_ethnicity = np.random.choice(choose_ethnicity, 1)[0]
                    description_str = f"A photograph of a person's face that meets the following requirements, gender: {generated_gender}, age: {generated_age} years old, ethnicity: {generated_ethnicity}"
            elif ta == 'gender':
                gender = filename.split('_')[1]
                if gender == '0': generated_gender = 'male'
                else: generated_gender = 'female'
                assert sa == 'ethnicity', 'when the target attribute is gender, the sensitive attribute must be ethnicity, detailed in ./configs/reasonableness_judgment function'

                generated_age = filename.split('_')[0]
                if filename.split('_')[2] != '0':
                    generated_ethnicity = 'White'
                    description_str = f"A photograph of a person's face that meets the following requirements, gender: {generated_gender}, age: {generated_age} years old, ethnicity: {generated_ethnicity}"
                else:
                    choose_ethnicity = ['Black', 'Asian', 'Indian', 'Hispanic', 'Middle Eastern']
                    generated_ethnicity = np.random.choice(choose_ethnicity, 1)[0]
                    description_str = f"A photograph of a person's face that meets the following requirements, gender: {generated_gender}, age: {generated_age} years old, ethnicity: {generated_ethnicity}"
            elif ta == 'ethnicity':
                ethnicity = filename.split('_')[2]
                if ethnicity == '0': generated_ethnicity = 'White'
                elif ethnicity == '1': generated_ethnicity = 'Black'
                elif ethnicity == '2': generated_ethnicity = 'Asian'
                elif ethnicity == '3': generated_ethnicity = 'Indian'
                else: generated_ethnicity = 'Middle Eastern'
                assert sa == 'age', 'when the target attribute is ethnicity, the sensitive attribute must be age, detailed in ./configs/reasonableness_judgment function'

                gender = filename.split('_')[1]
                if gender == '0': generated_gender ='male'
                else: generated_gender = 'female'
                if int(filename.split('_')[0]) < 35:
                    # over 40 years old and under 30 years old, the aim is to enlarge the age difference and improve the generalization ability of the model
                    generated_age = np.random.randint(40, 100)
                    description_str = f"A photograph of a person's face that meets the following requirements, gender: {generated_gender}, age: {generated_age} years old, ethnicity: {generated_ethnicity}"
                else:
                    generated_age = np.random.randint(0, 30)
                    description_str = f"A photograph of a person's face that meets the following requirements, gender: {generated_gender}, age: {generated_age}, ethnicity: {generated_ethnicity}"

            w_age = str(generated_age)
            if generated_gender == 'female': w_gender = '1'
            else: w_gender = '0'
            if generated_ethnicity == 'White': w_ethnicity = '0'
            elif generated_ethnicity == 'Black': w_ethnicity = '1'
            elif generated_ethnicity == 'Asian': w_ethnicity = '2'
            elif generated_ethnicity == 'Indian': w_ethnicity = '3'
            else: w_ethnicity = '4'
            current_time = datetime.now()
            timestamp = current_time.strftime("%Y%m%d%H%M%S%f")
            f_name = f'{w_age}_{w_gender}_{w_ethnicity}'
            f_path = f'{saved_path}/{f_name}_{timestamp}.jpg'
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)
            image = pipe(
                description_str,
                num_inference_steps=60,
                guidance_scale=4.5,
            ).images[0]
            # 存储到saved_path文件夹下
            image.save(f_path)
        elif ds.startswith('celeba'):
            # 不同数字代表不同属性，需要的时候注意添加
            att_list = []
            with open(config['dataset_kwargs']['data_root'] + 'Anno/list_attr_celeba.txt', 'r') as f:
                reader = f.readlines()
                for line in reader:
                    att_list.append(line.split())
            attributes = att_list[1]
            img_info = att_list[2:]
            targets = attributes[ta-1]
            sensitive = attributes[sa-1]
            target_value = img_info[int(filename.split('.')[0])-1][ta]
            sensitive_value = img_info[int(filename.split('.')[0])-1][sa]
            # this description is for the target attribute is male and the sensitive attribute is attractive
            if target_value == '1' and sensitive_value == '1':
                description_str = "A photo of a male who is very unkempt and not at all attractive."
            elif target_value == '1' and sensitive_value == '-1':
                description_str = "A photo of a male who is very handsome and attractive."
            elif target_value == '-1' and sensitive_value == '1':
                description_str = "A photo of a female who is very unkempt and not at all attractive."
            elif target_value == '-1' and sensitive_value == '-1':
                description_str = "A photo of a female who is very beautiful and attractive."
            f_path = f'{saved_path}/aug_img_{filename}'
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)
            image = pipe(
                description_str,
                num_inference_steps=60,
                guidance_scale=4.5,
            ).images[0]
            image.save(f_path)

    # construct the relationship between the generated data and the original data
    ori_train_data = np.array(ori_img_list).tolist()
    generated_train_data = [f for f in os.listdir(saved_path)]
    generated_train_data.sort()
    ori_train_data.extend(generated_train_data)
    os.makedirs(f'./dataset/{ds}/{ds}_{ta}_{sa}_{nclass}/', exist_ok=True)
    torch.save(ori_train_data, f'./dataset/{ds}/{ds}_{ta}_{sa}_{nclass}/relations_{n_labeled_samples}.pt')

def annotate_files_use_qwen72b(file_path, ds):
    if ds.startswith('utkface'):
        content_str = """Describe the gender, age, and race of the person in the image. Gender should be chosen from \'male\' or \'female\'; age should be a positive integer; and race should be one of \'White\', \'Black\', \'Asian\', \'Indian\', or \'Other\'. Output your response strictly as a JSON object without any additional text, explanations, or formatting. The exact format must be: {"Gender": "male", "Age": "15", "Race": "Asian"}. Do not include quotation marks around the entire JSON, and do not provide any other details or comments."""
    elif ds.startswith('celeba'):
        content_str = """Describe the gender and attractive of the person in the image. Gender should be chosen from \'male\' or \'female\'; attractive should be chosen from \'yes\' or \'no\'. Output your response strictly as a JSON object without any additional text, explanations, or formatting. The exact format must be: {"Gender": "male", "Attractive": "yes"}. Do not include quotation marks around the entire JSON, and do not provide any other details or comments."""
    else:
        assert False, 'Invalid dataset'
    #  base 64 编码格式
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    base64_image = encode_image(file_path)
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key="sk-XXX",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    try:
        completion = client.chat.completions.create(
            model="qwen2.5-vl-72b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            # 需要注意，传入BASE64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                            # PNG图像：  f"data:image/png;base64,{base64_image}"
                            # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                            # WEBP图像： f"data:image/webp;base64,{base64_image}"
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                        {"type": "text", "text": content_str},
                    ],
                }
            ],
            temperature=0.01,
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content
    except Exception as e:
        return int(0)

def get_age_length(data):
    json_data = json.loads(data[0])
    return len(json_data['Age'])

def annotate_files_with_json(data):
    json_data = json.loads(data)
    age = json_data['Age']
    race = json_data['Race']
    gender = json_data['Gender']
    if gender == "female":
        gender_str = "1"
    elif gender == "male":
        gender_str = "0"
    else:
        assert False, "Gender should be 'female' or 'male'"
    if race == "White":
        race_str = "0"
    elif race == "Black":
        race_str = "1"
    elif race == "Asian":
        race_str = "2"
    elif race == "Indian":
        race_str = "3"
    elif race == "Other":
        race_str = "4"
    else:
        assert False, "Race should be 'White', 'Black', 'Asian', 'Indian', or 'Other'"
    age_str = str(age)
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d%H%M%S%f")
    f_name = f'{age_str}_{gender_str}_{race_str}_{timestamp}.jpg'
    return f_name

def get_unlabeled_files(train_data_path, databse_data_path, config):
    train_date = torch.load(train_data_path)
    database_date = torch.load(databse_data_path)
    mask = np.isin(database_date, train_date)
    unlabeled_samples_ids = database_date[~mask]
    if config['dataset'].startswith('celeba'):
        data_root = os.path.join(config['dataset_kwargs']['data_root'], 'Img/img_align_celeba')
    else:
        data_root = config['dataset_kwargs']['data_root']
    files_all = os.listdir(data_root)
    files_all.sort()
    unlabeled_samples_files = np.array(files_all)[unlabeled_samples_ids].tolist()
    return unlabeled_samples_files

def get_low_confidence_files(unlabeled_samples, threshold, config):
    low_confidence_files = []
    ds = config['dataset']
    process_path = f'./processed_files/{ds}/image_probs_sorted_dict.pt'
    os.makedirs(f'./processed_files/{ds}', exist_ok=True)
    if os.path.exists(process_path):
        image_probs_sorted_dict = torch.load(process_path)
        sorted_probs = torch.stack([image_probs_sorted_dict[i] for i in unlabeled_samples], dim=0)
    else:
        # generate all Prompts
        cach_dir = "./LLM/clip"
        model = AutoModel.from_pretrained('jinaai/jina-clip-v2', cache_dir=cach_dir, trust_remote_code=True,local_files_only=True)
        model = model.to('cuda:0')
        if ds.startswith('utkface'):
            age_groups = [
                ("young child (0-20 years old)", 0),
                ("young adult (20-40 years old)", 1),
                ("middle-aged adult (40-60 years old)", 2),
                ("elderly person (60-80 years old)", 3),
                ("very old person (80+ years old)", 4)
            ]

            races = ["White", "Black", "Asian", "Indian", "Other"]
            genders = ["male", "female"]
            class_descriptions = []
            for age_desc, age_id in age_groups:
                for race in races:
                    for gender in genders:
                        general_description = f"A close-up portrait of a {age_desc} {race} {gender}, looking directly at the camera, with a neutral expression, in sharp focus, natural lighting, high resolution, detailed facial features"
                        class_descriptions.append(general_description)

            text_embeddings = model.encode_text(class_descriptions)
            path = config['dataset_kwargs']['data_root']
        elif ds.startswith('celeba'):
            ta = config['dataset_kwargs']['target_attribute']
            sa = config['dataset_kwargs']['sensitive_attribute']
            if ta == 3 and sa == 21:
                class_descriptions = ['a male who is very unkempt and not at all attractive.',
                                      'a female who is very beautiful and attractive.',
                                      'a female who is very unkempt and not at all attractive.',
                                      'a female who is very handsome and attractive.']
                text_embeddings = model.encode_text(class_descriptions)
                path = config['dataset_kwargs']['data_root']+'Img/img_align_celeba'
            else:
                assert False, f'{ta} and {sa} are not supported in celeba dataset, please add them to the code'
        else:
            assert False, "Dataset not supported"


        files = [os.path.join(path, f) for f in unlabeled_samples]
        truncate_dim = 1024
        image_embeddings = model.encode_image(files, truncate_dim=truncate_dim)  # also accepts PIL.Image.Image, local filenames, dataURI
        logits = (image_embeddings @ text_embeddings.T) / 0.01  # temperature of 0.01
        probs = torch.tensor(logits).softmax(dim=-1)
        sorted_probs = torch.sort(probs, descending=True)[0]
    mask = sorted_probs[:, 0] < threshold
    assert len(unlabeled_samples) == len(mask)
    low_confidence_files.extend(np.array(unlabeled_samples)[mask])

    return low_confidence_files

def get_high_confidence_files_for_utkface(low_confidence_files, original_img_files, config, labeled_LLM='Qwen_2_5_7B', low_threshold=0.2, high_threshold=0.9):

    cache_dir = "./LLM/QWen2-VL"
    model_dir = snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=cache_dir, local_files_only=True)

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )

    # default processer
    processor = AutoProcessor.from_pretrained(model_dir)

    path = config['dataset_kwargs']['data_root']
    ds = config['dataset']
    ta = config['dataset_kwargs']['target_attribute']
    sa = config['dataset_kwargs']['sensitive_attribute']
    nclass = config['arch_kwargs']['nclass']
    high_confidence_files = []
    n_labeled_samples = config['sample']['labeled_sample_num']
    str_low_threshold = str(low_threshold).replace('0.','')
    str_high_threshold = str(high_threshold).replace('0.','')
    saved_path = f'./dataset/pseudo_data/{ds}_{ta}_{sa}_{nclass}/samples_{n_labeled_samples}_{str_low_threshold}_{str_high_threshold}'
    if os.path.exists(saved_path):
        shutil.rmtree(saved_path)

    process_image_path = f'./processed_files/{ds}/image_next_token_probs.pt'
    if os.path.exists(process_image_path):
        image_next_token_probs = torch.load(process_image_path)
        processed_images = list(image_next_token_probs.keys())
    else:
        image_next_token_probs = {}
        processed_images = []

    for file in tqdm(low_confidence_files, desc='Annotating high confidence files'):
        each_item_logits = {}
        image_path = os.path.join(path, file)
        if file in processed_images:
            # obtain the logits for each item
            if config['dataset_kwargs']['target_attribute'] == 'age':
                logits_all = image_next_token_probs[file]['top1_logits_age_avg']
            elif config['dataset_kwargs']['target_attribute'] == 'ethnicity':
                logits_all = image_next_token_probs[file]['top1_logits_race']
            elif config['dataset_kwargs']['target_attribute'] == 'gender':
                logits_all = image_next_token_probs[file]['top1_logits_gender']
            else:
                # easy for gender, abandon it
                logits_all = image_next_token_probs[file]['logits_age_race_avg']
            if logits_all > high_threshold:
                if 'qwen_72b_filenames' in image_next_token_probs[file]:
                    annotated_file_name = image_next_token_probs[file]['qwen_72b_filenames']
                else:
                    qweb_72b_preds = annotate_files_use_qwen72b(image_path, ds)
                    if qweb_72b_preds == 0:
                        annotated_file_name = file
                    else:
                        annotated_file_name = annotate_files_with_json(qweb_72b_preds)
                    image_next_token_probs[file]['qwen_72b_preds'] = qweb_72b_preds
                    image_next_token_probs[file]['qwen_72b_filenames'] = annotated_file_name
                    torch.save(image_next_token_probs, process_image_path)
                os.makedirs(saved_path, exist_ok=True)
                shutil.copy(image_path, os.path.join(saved_path, annotated_file_name))
                high_confidence_files.append(annotated_file_name)
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"file://{image_path}",
                        },
                        {"type": "text",
                         "text": """
                    Describe the gender, age, and race of the person in the image. Gender should be chosen from 'male' or 'female'; age should be a positive integer; and race should be one of 'White', 'Black', 'Asian', 'Indian', or 'Other'. Output your response strictly as a JSON object without any additional text, explanations, or formatting. The exact format must be: {"Gender": "male", "Age": "31", "Race": "Asian"}. Do not include quotation marks around the entire JSON, and do not provide any other details or comments.
                    """},
                    ],
                }
            ]
            # Preparation for inference
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(
                **inputs,
                output_scores=True,
                output_logits=True,
                max_new_tokens=128,
                return_dict_in_generate=True,
                temperature=0.01)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids.sequences)]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            age_length = get_age_length(output_text)
            logits_gender, logits_age,  logits_race = 0.0, 0.0, 0.0
            for step, (score, logit) in enumerate(zip(generated_ids.scores, generated_ids.logits)):
                output_token = processor.batch_decode([generated_ids_trimmed[0][step]], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                # 4, 10+, -3
                probabilities_logit = torch.softmax(logit, dim=1)
                top_probabilities_logit, top2_indices_logit = torch.topk(probabilities_logit, k=1, dim=1)

                if step == 4 and output_token[0] in ['male', 'female']:
                    logits_gender += top_probabilities_logit[0, 0]
                elif step in range(10, 10+age_length) and (output_token[0].isdigit()): #  or output_token[0] in ['0', '2', '-', '4', '6', '8'] or output_token[0].startswith('+')
                    logits_age += top_probabilities_logit[0, 0]
                elif step == len(generated_ids.scores)-3 and output_token[0] in ['White', 'Black', 'Asian', 'Indian', 'Other']:
                    logits_race += top_probabilities_logit[0, 0]
            if logits_gender == 0.0 or logits_age == 0.0 or logits_race == 0.0:
                continue
            # obtain the final logits
            if config['dataset_kwargs']['target_attribute'] == 'age':
                logits_all = logits_age / (age_length)
            elif config['dataset_kwargs']['target_attribute'] == 'ethnicity':
                logits_all = logits_race
            elif config['dataset_kwargs']['target_attribute'] == 'gender':
                logits_all = logits_gender
            else:
                logits_all = (logits_age / (age_length) + logits_race) / 2

            each_item_logits['top1_logits_gender'] = logits_gender
            each_item_logits['top1_logits_age_avg'] = logits_age / (age_length)
            each_item_logits['top1_logits_race'] = logits_race

            each_item_logits['logits_age_race_avg'] = (logits_age / (age_length) + logits_race) / 2
            each_item_logits['logits_age_race_gender_avg'] = (logits_gender + logits_age / (age_length) + logits_race) / 3

            qweb_7b_preds = output_text[0]
            each_item_logits['qwen_7b_preds'] = qweb_7b_preds
            each_item_logits['qwen_7b_filenames'] = annotate_files_with_json(qweb_7b_preds)
            if logits_all > high_threshold:
                qweb_72b_preds = annotate_files_use_qwen72b(image_path, ds)
                if qweb_72b_preds == 0:
                    annotated_file_name = file
                else:
                    annotated_file_name = annotate_files_with_json(qweb_72b_preds)
                each_item_logits['qwen_72b_preds'] = qweb_72b_preds
                each_item_logits['qwen_72b_filenames'] = annotated_file_name

                # if labeled_LLM == 'Qwen_2_5_7B':
                #     json_output_text = output_text[0]
                # elif labeled_LLM == 'Qwen_2_5_72B':
                #     json_output_text = annotate_files_use_qwen72b(image_path)
                # else:
                #     assert False, "labeled_LLM should be 'Qwen_2_5_7B' or 'Qwen_2_5_72B'"
                os.makedirs(saved_path, exist_ok=True)
                shutil.copy(image_path, os.path.join(saved_path, annotated_file_name))
                high_confidence_files.append(annotated_file_name)
            image_next_token_probs[file] = each_item_logits
            torch.save(image_next_token_probs, process_image_path)

    # construct the relationship between the generated data and the original data
    ori_train_data = np.array(original_img_files).tolist()
    high_confidence_files.sort()
    ori_train_data.extend(high_confidence_files)
    os.makedirs(f'./dataset/{ds}/{ds}_{ta}_{sa}_{nclass}', exist_ok=True)
    torch.save(ori_train_data, f'./dataset/{ds}/{ds}_{ta}_{sa}_{nclass}/pseudo_relations_{n_labeled_samples}_{str_low_threshold}_{str_high_threshold}.pt')
    return high_confidence_files

def get_high_confidence_files_for_celeba(low_confidence_files, original_img_files, config, labeled_LLM='Qwen_2_5_7B', low_threshold=0.2, high_threshold=0.9):

    cache_dir = "./LLM/QWen2-VL"
    model_dir = snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=cache_dir, local_files_only=True)

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )

    # default processer
    processor = AutoProcessor.from_pretrained(model_dir)

    path = config['dataset_kwargs']['data_root'] + 'Img/img_align_celeba'
    ds = config['dataset']
    ta = config['dataset_kwargs']['target_attribute']
    sa = config['dataset_kwargs']['sensitive_attribute']
    assert ta == 3 and sa == 21, f'{ta} and {sa} are not supported in celeba dataset, please add them to the code'
    nclass = config['arch_kwargs']['nclass']
    high_confidence_files = []
    n_labeled_samples = config['sample']['labeled_sample_num']
    str_low_threshold = str(low_threshold).replace('0.','')
    str_high_threshold = str(high_threshold).replace('0.','')
    saved_path = f'./dataset/pseudo_data/{ds}_{ta}_{sa}_{nclass}/samples_{n_labeled_samples}_{str_low_threshold}_{str_high_threshold}'
    if os.path.exists(saved_path):
        shutil.rmtree(saved_path)

    process_image_path = f'./processed_files/{ds}/image_next_token_probs.pt'
    if os.path.exists(process_image_path):
        image_next_token_probs = torch.load(process_image_path)
        processed_images = list(image_next_token_probs.keys())
    else:
        image_next_token_probs = {}
        processed_images = []

    for file in tqdm(low_confidence_files, desc='Annotating high confidence files'):
        each_item_logits = {}
        image_path = os.path.join(path, file)
        if file in processed_images:
            if config['dataset_kwargs']['target_attribute'] == 3:
                logits_all = image_next_token_probs[file]['top1_logits_attractive']
            elif config['dataset_kwargs']['target_attribute'] == 21:
                logits_all = image_next_token_probs[file]['top1_logits_gender']
            else:
                logits_all = image_next_token_probs[file]['logits_gender_attractive_avg']

            if logits_all > high_threshold:
                if 'qwen_72b_filenames' in image_next_token_probs[file]:
                    annotated_file_name = image_next_token_probs[file]['qwen_72b_filenames']
                else:
                    qweb_72b_preds = annotate_files_use_qwen72b(image_path, ds)
                    if qweb_72b_preds == 0:
                        annotated_file_name = file
                    else:
                        annotated_file_name = f'ann_img_{file}'
                    image_next_token_probs[file]['qwen_72b_preds'] = qweb_72b_preds
                    image_next_token_probs[file]['qwen_72b_filenames'] = annotated_file_name
                    torch.save(image_next_token_probs, process_image_path)
                os.makedirs(saved_path, exist_ok=True)
                shutil.copy(image_path, os.path.join(saved_path, annotated_file_name))
                high_confidence_files.append(annotated_file_name)
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"file://{image_path}",
                        },
                        {"type": "text",
                         "text": """
                         Describe the gender and attractive of the person in the image. Gender should be chosen from 'male' or 'female'; attractive should be chosen from 'yes' or 'no'. Output your response strictly as a JSON object without any additional text, explanations, or formatting. The exact format must be: {"Gender": "male", "Attractive": "yes"}. Do not include quotation marks around the entire JSON, and do not provide any other details or comments.
                         """},
                    ],
                }
            ]
            # Preparation for inference
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(
                **inputs,
                output_scores=True,
                output_logits=True,
                max_new_tokens=128,
                return_dict_in_generate=True,
                temperature=0.01)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids.sequences)]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            logits_gender, logits_attractive = 0.0, 0.0
            for step, (score, logit) in enumerate(zip(generated_ids.scores, generated_ids.logits)):
                output_token = processor.batch_decode([generated_ids_trimmed[0][step]], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                # 4, 10+, -3
                probabilities_logit = torch.softmax(logit, dim=1)
                top_probabilities_logit, top2_indices_logit = torch.topk(probabilities_logit, k=1, dim=1)

                if step == 4 and output_token[0] in ['male', 'female']:
                    logits_gender += top_probabilities_logit[0, 0]
                elif step == 11 and output_token[0] in ['yes', 'no']:
                    logits_attractive += top_probabilities_logit[0, 0]
            if logits_gender == 0.0 or logits_attractive == 0.0:
                continue

            if config['dataset_kwargs']['target_attribute'] == 3:
                logits_all = logits_attractive
            elif config['dataset_kwargs']['target_attribute'] == 21:
                logits_all = logits_gender
            else:
                logits_all = (logits_gender + logits_attractive) / 2

            each_item_logits['top1_logits_gender'] = logits_gender
            each_item_logits['top1_logits_attractive'] = logits_attractive
            each_item_logits['logits_gender_attractive_avg'] = (logits_gender + logits_attractive) / 2

            qweb_7b_preds = output_text[0]
            each_item_logits['qwen_7b_preds'] = qweb_7b_preds
            each_item_logits['qwen_7b_filenames'] = f'ann_img_{file}'
            if logits_all > high_threshold:
                qweb_72b_preds = annotate_files_use_qwen72b(image_path, ds)
                if qweb_72b_preds == 0:
                    annotated_file_name = file
                else:
                    annotated_file_name = f'ann_img_{file}'
                each_item_logits['qwen_72b_preds'] = qweb_72b_preds
                each_item_logits['qwen_72b_filenames'] = annotated_file_name

                os.makedirs(saved_path, exist_ok=True)
                shutil.copy(image_path, os.path.join(saved_path, annotated_file_name))
                high_confidence_files.append(annotated_file_name)
            image_next_token_probs[file] = each_item_logits
            torch.save(image_next_token_probs, process_image_path)

    # construct the relationship between the generated data and the original data
    ori_train_data = np.array(original_img_files).tolist()
    high_confidence_files.sort()
    ori_train_data.extend(high_confidence_files)
    os.makedirs(f'./dataset/{ds}/{ds}_{ta}_{sa}_{nclass}', exist_ok=True)
    torch.save(ori_train_data, f'./dataset/{ds}/{ds}_{ta}_{sa}_{nclass}/pseudo_relations_{n_labeled_samples}_{str_low_threshold}_{str_high_threshold}.pt')
    return high_confidence_files
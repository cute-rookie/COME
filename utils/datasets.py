import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets.folder import pil_loader
from torchvision import transforms
from PIL import Image
from utils.tools import get_labeled_data, get_relations, generate_samples_use_cdm, get_att_in_celeba
from utils.tools import get_unlabeled_files, get_low_confidence_files, get_high_confidence_files_for_utkface, get_high_confidence_files_for_celeba


class HashingDataset(Dataset):
    def __init__(self, root,
                 transform=None,
                 target_transform=None,
                 filename='train',
                 separate_multiclass=False):
        self.loader = pil_loader
        self.separate_multiclass = separate_multiclass
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.filename = filename
        self.train_data = []
        self.train_labels = []

        filename = os.path.join(self.root, self.filename)

        with open(filename, 'r') as f:
            while True:
                lines = f.readline()
                if not lines:
                    break

                path_tmp = lines.split()[0]
                label_tmp = lines.split()[1:]
                self.is_onehot = len(label_tmp) != 1
                if not self.is_onehot:
                    label_tmp = lines.split()[1]
                if self.separate_multiclass:
                    assert self.is_onehot, 'if multiclass, please use onehot'
                    nonzero_index = np.nonzero(np.array(label_tmp, dtype=np.int))[0]
                    for c in nonzero_index:
                        self.train_data.append(path_tmp)
                        label_tmp = ['1' if i == c else '0' for i in range(len(label_tmp))]
                        self.train_labels.append(label_tmp)
                else:
                    self.train_data.append(path_tmp)
                    self.train_labels.append(label_tmp)

        self.train_data = np.array(self.train_data)
        self.train_labels = np.array(self.train_labels, dtype=np.float)

        print(f'Number of data: {self.train_data.shape[0]}')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]
        target = torch.tensor(target)

        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.train_data)


def one_hot(nclass):
    def f(index):
        index = torch.tensor(int(index)).long()
        return torch.nn.functional.one_hot(index, nclass)

    return f


class UTKface(Dataset):
    def __init__(self, nclass, ta, sa, config, transform):
        self.data_folder = config['dataset_kwargs']['data_root']
        n_labeled_samples = config['sample']['labeled_sample_num']
        str_low_threshold = str(config['low_confidence_threshold']).replace('0.', '')
        str_high_threshold = str(config['high_confidence_threshold']).replace('0.', '')
        self.generated_image_folder = os.path.join(config['dataset_kwargs']['generated_image_root'],
                                                   f'samples_{n_labeled_samples}')
        self.pseudo_image_folder = os.path.join(config['dataset_kwargs']['pseudo_image_root'],
                                                f'samples_{n_labeled_samples}_{str_low_threshold}_{str_high_threshold}')
        self.size = 224
        self.img_list = os.listdir(self.data_folder)
        self.img_list.sort()
        self.transform = transform
        self.att = []
        self.ethnicity_list = []
        self.age_list = []
        self.gender_list = []
        self.ta = ta
        self.sa = sa
        self.data = []
        self.nclass = nclass
        class_type = config['class_type']
        self.class_type = class_type

        for i in range(len(self.img_list)):
            self.gender_list.append(int(self.img_list[i].split('_')[1] == '0'))
            if class_type == 0:
                self.age_list.append(int(self.img_list[i].split('_')[0]) < 35)
                self.ethnicity_list.append(int(self.img_list[i].split('_')[2] == '0'))
            elif class_type == 1:
                self.age_list.append(int(self.img_list[i].split('_')[0]) < 35)
                self.ethnicity_list.append(int(self.img_list[i].split('_')[2]))
            elif class_type == 2:
                self.ethnicity_list.append(int(self.img_list[i].split('_')[2] == '0'))
                if int(self.img_list[i].split('_')[0]) < 20:
                    self.age_list.append(0)
                elif int(self.img_list[i].split('_')[0]) < 40:
                    self.age_list.append(1)
                elif int(self.img_list[i].split('_')[0]) < 60:
                    self.age_list.append(2)
                elif int(self.img_list[i].split('_')[0]) < 80:
                    self.age_list.append(3)
                else:
                    self.age_list.append(4)

        self.age_list = np.array(self.age_list).flatten()
        self.ethnicity_list = np.array(self.ethnicity_list).flatten()
        self.gender_list = np.array(self.gender_list).flatten()
        self.img_list = np.array(self.img_list).flatten()

    def ratio_preprocess(self,
                         alpha=2):  # the sensitve group (ethnicity) has male data alhpha tiems as much as female data
        male = []
        female = []
        data_inedx = []
        for i in range(len(self.img_list)):  #
            if self.ethnicity_list[i] == 1:
                if self.gender_list[i] == 0:
                    male.append(i)
                else:
                    female.append(i)
            else:
                data_inedx.append(i)
        ratio_female = female[:len(male) // alpha]
        data_inedx = data_inedx + male + ratio_female
        self.img_list = self.img_list[data_inedx]
        self.age_list = self.age_list[data_inedx]
        self.ethnicity_list = self.ethnicity_list[data_inedx]
        self.gender_list = self.gender_list[data_inedx]

        print('male nums in ethnicity: ', len(male))
        print('female nums in ethnicity: ', len(ratio_female))
        print('ratio data: ', len(self.img_list))

    def __getitem__(self, index1):

        # index2=random.choice(range(len(self.img_list)))
        age = int(self.age_list[index1])
        gender = int(self.gender_list[index1])
        ethnicity = int(self.ethnicity_list[index1])

        age = torch.from_numpy(np.array(age)).long()
        gender = torch.from_numpy(np.array(gender)).long()
        ethnicity = torch.from_numpy(np.array(ethnicity)).long()

        gender = torch.nn.functional.one_hot(gender, 2)
        if self.class_type == 0:
            age = torch.nn.functional.one_hot(age, 2)
            ethnicity = torch.nn.functional.one_hot(ethnicity, 2)
        elif self.class_type == 1:
            age = torch.nn.functional.one_hot(age, 2)
            ethnicity = torch.nn.functional.one_hot(ethnicity, self.nclass)
        elif self.class_type == 2:
            age = torch.nn.functional.one_hot(age, self.nclass)
            ethnicity = torch.nn.functional.one_hot(ethnicity, 2)

        ta = 0
        sa = 0

        if os.path.exists(os.path.join(self.data_folder, self.img_list[index1])):
            img1 = Image.open(os.path.join(self.data_folder, self.img_list[index1])).convert('RGB')
            # img2=Image.open(self.data_folder+self.img_list[index2])
        elif os.path.exists(os.path.join(self.generated_image_folder, self.img_list[index1])):
            img1 = Image.open(os.path.join(self.generated_image_folder, self.img_list[index1])).convert('RGB')
        elif os.path.exists(os.path.join(self.pseudo_image_folder, self.img_list[index1])):
            img1 = Image.open(os.path.join(self.pseudo_image_folder, self.img_list[index1])).convert('RGB')
        else:
            assert False, f'{self.img_list[index1]} not found'

        if self.ta == 'gender':
            ta = gender
        elif self.ta == 'age':
            ta = age
        elif self.ta == 'ethnicity':
            ta = ethnicity

        if self.sa == "gender":
            sa = gender
        elif self.sa == "age":
            sa = age
        elif self.sa == "ethnicity":
            sa = ethnicity

        return self.transform(img1), ta, sa

    def __len__(self):
        return (self.img_list.shape[0])


class Celeba(Dataset):
    def __init__(self, ta, ta2, sa, sa2, config, transform):
        self.data_folder = config['dataset_kwargs']['data_root']
        self.img_list = os.listdir(self.data_folder + 'Img/img_align_celeba/')
        n_labeled_samples = config['sample']['labeled_sample_num']
        str_low_threshold = str(config['low_confidence_threshold']).replace('0.', '')
        str_high_threshold = str(config['high_confidence_threshold']).replace('0.', '')
        self.generated_image_folder = os.path.join(config['dataset_kwargs']['generated_image_root'],
                                                   f'samples_{n_labeled_samples}')
        self.pseudo_image_folder = os.path.join(config['dataset_kwargs']['pseudo_image_root'],
                                                f'samples_{n_labeled_samples}_{str_low_threshold}_{str_high_threshold}')

        self.img_list.sort()
        self.transform = transform
        self.att = []

        att_list = []
        eval_list = []
        with open(self.data_folder + 'Anno/list_attr_celeba.txt', 'r') as f:
            reader = f.readlines()
            for line in reader:
                att_list.append(line.split())
        att_list = att_list[2:]

        with open(self.data_folder + 'Eval/list_eval_partition.txt', 'r') as f:
            reader = f.readlines()
            for line in reader:
                eval_list.append(line.split())

        for i, eval_inst in enumerate(eval_list):
            # if eval_inst[1]==str(self.split):
            if att_list[i][0] == eval_inst[0]:
                self.att.append(att_list[i])
            else:
                pass

        self.att = np.array(self.att)
        self.att = (self.att == '1').astype(int)
        self.img_list = np.array(self.img_list)
        self.ta = ta
        self.ta2 = ta2
        self.sa = sa
        self.sa2 = sa2
        self.nclass = 2

    def __getitem__(self, index1):

        ta = self.att[index1][int(self.ta)]
        sa = self.att[index1][int(self.sa)]

        if self.ta2 != 'None':
            ta2 = self.att[index1][int(self.ta2)]
            ta = ta + 2 * ta2

        if self.sa2 != 'None':
            sa2 = self.att[index1][int(self.sa2)]
            sa = sa + 2 * sa2

        ta = torch.nn.functional.one_hot(torch.from_numpy(np.array(ta)), self.nclass)
        sa = torch.nn.functional.one_hot(torch.from_numpy(np.array(sa)), self.nclass)

        # index2=random.choice(range(len(self.img_list)))
        if os.path.exists(self.data_folder + 'Img/img_align_celeba/' + self.img_list[index1]):
            img1 = Image.open(os.path.join(self.data_folder + 'Img/img_align_celeba/', self.img_list[index1])).convert('RGB')
        elif os.path.exists(os.path.join(self.generated_image_folder, self.img_list[index1])):
            img1 = Image.open(os.path.join(self.generated_image_folder, self.img_list[index1])).convert('RGB')
        elif os.path.exists(os.path.join(self.pseudo_image_folder, self.img_list[index1])):
            img1 = Image.open(os.path.join(self.pseudo_image_folder, self.img_list[index1])).convert('RGB')
        else:
            assert False, f'{self.img_list[index1]} not found'

        # img2=Image.open(self.data_folder+'Img/img_align_celeba/'+self.img_list[index2])

        return self.transform(img1), ta, sa

    def __len__(self):
        return len(self.att)


def celeba(**kwargs):

    transform = kwargs['transform']
    filename = kwargs['filename']
    reset = kwargs['reset']
    config = kwargs['config']
    transform_mode = kwargs['transform_mode']

    nclass = config['arch_kwargs']['nclass']
    n_labeled_samples = config['sample']['labeled_sample_num']
    ta = config['dataset_kwargs']['target_attribute']
    sa = config['dataset_kwargs']['sensitive_attribute']
    class_type = config['class_type']
    ds = config['dataset']

    nclass = 2
    transform = kwargs['transform']
    fn = kwargs['filename']
    reset = kwargs['reset']

    celeba = Celeba(ta=ta, ta2='None', sa=sa, sa2='None', config=config, transform=transform)

    data = celeba.img_list
    att = celeba.att
    targets = att[:, ta]

    # obtain labeled samples
    data_index, generate_data = get_labeled_data(filename, reset, nclass, n_labeled_samples, targets, ds, ta, sa)

    data = np.array(celeba.img_list)
    att = np.array(celeba.att)

    all_img_list = data.copy()
    all_att_list = att.copy()

    celeba.img_list = data[data_index]
    celeba.att = att[data_index]

    original_img_files = celeba.img_list.copy()
    original_att = celeba.att.copy()

    ################### diffusion model to generate labeled samples, with same target attribute and different sensitive attribute #######################
    # 是否使用增强的数据
    if config['aug_for_labeled_sample'] and transform_mode == 'train':
        # 首先查看relation.pt文件，里面记录的是标签样本与增强样本的文件名，看看在文件夹中是否对应，不对应的话，重新生成
        relation_path = f'./dataset/{ds}/{ds}_{ta}_{sa}_{nclass}/relations_{n_labeled_samples}.pt'
        sample_path = f'./dataset/generated_data/{ds}_{ta}_{sa}_{nclass}/samples_{n_labeled_samples}'
        relations = get_relations(relation_path, sample_path, original_img_files)  # 返回False说明train.txt中的样本与存在的生成样本不匹配，需要重新生成
        if config['sample']['generate_labeled_sample'] or generate_data or reset or not relations:
            # generate_data表示是否重新生成增强数据，返回True则重新生成增强数据，那么对应的图像也应该是重新生成的
            # save the generated samples to the ./generated_data folder
            print('Generating labeled samples using conditional diffusion model...')
            generate_samples_use_cdm(celeba.img_list, ta, sa, config)
            assert False, 'Please restart the program to load the generated data.'
        # 将增强之后的数据也添加到celeba数据集中
        generated_image_path = f'./dataset/generated_data/{ds}_{ta}_{sa}_{nclass}/samples_{n_labeled_samples}'
        assert os.path.exists(generated_image_path), f'The generated data path {generated_image_path} does not exist.'
        generated_img_list = []
        generated_att = []
        for img_name in os.listdir(generated_image_path):
            generated_img_list.append(img_name)
            id = int(img_name.split('_')[-1].split('.')[0])
            file_att = all_att_list[id-1]
            file_att[sa] = 1 - file_att[sa]
            generated_att.append(file_att)

        generated_img_list = np.array(generated_img_list)
        generated_att = np.array(generated_att)

        celeba.img_list = np.concatenate((celeba.img_list, generated_img_list))
        celeba.att = np.concatenate((celeba.att, generated_att))

    ################### end of diffusion model ##########

    ################### annotate unlabeled samples use LLM ###################
    if config['annotation_unlabeled_data'] and transform_mode == 'train':
        # 首先查看relation.pt文件，里面记录的是标签样本与增强样本的文件名，看看在文件夹中是否对应，不对应的话，重新生成
        low_threshold = config['low_confidence_threshold']
        high_threshold = config['high_confidence_threshold']
        str_low_threshold = str(low_threshold).replace('0.', '')
        str_high_threshold = str(high_threshold).replace('0.', '')
        relation_path = f'./dataset/{ds}/{ds}_{ta}_{sa}_{nclass}/pseudo_relations_{n_labeled_samples}_{str_low_threshold}_{str_high_threshold}.pt'
        sample_path = f'./dataset/pseudo_data/{ds}_{ta}_{sa}_{nclass}/samples_{n_labeled_samples}_{str_low_threshold}_{str_high_threshold}'
        relations = get_relations(relation_path, sample_path, original_img_files)  # 返回False说明train.txt中的样本与存在的生成样本不匹配，需要重新生成
        if not relations:
            print('Annotating unlabeled samples using LLM...')
            # 获取全部未标记数据
            train_ids_path = f'./dataset/{ds}/{ds}_{ta}_{sa}_{nclass}/{filename}'
            database_ids_path = f'./dataset/{ds}/{ds}_{ta}_{sa}_{nclass}/database_{n_labeled_samples}.txt'
            unlabeled_data_files = get_unlabeled_files(train_ids_path, database_ids_path, config)
            # test_unlabeled_data_files = random.sample(unlabeled_data_files, 10)  # 随机选取100个未标记数据进行测试
            # 采用多模态表征来找到low_confidence的样本
            low_confidence_files = get_low_confidence_files(unlabeled_data_files, config['low_confidence_threshold'], config)

            # 采用LLM方法对low_confidence的样本进行处理，获取high_confidence的样本
            labeled_LLM = config['labeled_LLM']
            high_confidence_files = get_high_confidence_files_for_celeba(low_confidence_files, original_img_files, config, labeled_LLM, low_threshold, high_threshold)
        else:
            # 不需要重新标注，直接读取
            high_confidence_files = os.listdir(sample_path)
        # 更新celeba数据集
        if len(high_confidence_files) != 0:
            image_next_token_probs = torch.load('./processed_files/celeba/image_next_token_probs.pt')
            pseudo_img_list = []
            pseudo_att = []
            for img_name in high_confidence_files:
                pseudo_img_list.append(img_name)
                id = int(img_name.split('_')[-1].split('.')[0])
                file_att = all_att_list[id-1]
                ori_file_name = img_name.split('_')[-1]
                pred_info = image_next_token_probs[ori_file_name]['qwen_72b_preds']
                gender, attractiveness = get_att_in_celeba(pred_info)
                file_att[ta] = attractiveness
                file_att[sa] = gender
                pseudo_att.append(file_att)
            pseudo_img_list = np.array(pseudo_img_list)
            pseudo_att = np.array(pseudo_att)

            celeba.img_list = np.concatenate((celeba.img_list, pseudo_img_list))
            celeba.att = np.concatenate((celeba.att, pseudo_att))
    ################### end of annotate unlabeled samples use LLM ###################

    return celeba


def utk(**kwargs):
    nclass = 2
    transform = kwargs['transform']
    fn = kwargs['filename']
    reset = kwargs['reset']

    utkface = UTKface(nclass=2, ta='gender', sa='ethnicity', data_folder='/home/zf/dataset/utkface/UTKFace/',
                      transform=transform, multiclass=0)
    # utkface.ratio_preprocess(alpha=4)

    data = utkface.img_list
    gender = utkface.gender_list
    age = utkface.age_list
    ethnicity = utkface.ethnicity_list
    targets = gender

    path = f'/home/zf/dataset/utkface/{fn}'

    load_data = fn == 'train.txt'
    load_data = load_data and (reset or not os.path.exists(path))

    if not load_data:
        print(f'Loading {path}')
        data_index = torch.load(path)
    else:
        train_data_index = []
        query_data_index = []
        db_data_index = []

        data_id = np.arange(data.shape[0])  # [0, 1, ...]

        for i in range(nclass):
            class_mask = targets == i
            index_of_class = data_id[class_mask].copy()  # index of the class [2, 10, 656,...]
            np.random.shuffle(index_of_class)

            query_n = 250  # // (nclass // 10)

            index_for_query = index_of_class[:query_n].tolist()
            index_for_db = index_of_class[query_n:].tolist()
            index_for_train = index_for_db

            train_data_index.extend(index_for_train)
            query_data_index.extend(index_for_query)
            db_data_index.extend(index_for_db)

        train_data_index = np.array(train_data_index)
        query_data_index = np.array(query_data_index)
        db_data_index = np.array(db_data_index)

        print('train_data_index', train_data_index.shape)
        print('query_data_index', query_data_index.shape)
        print('db_data_index', db_data_index.shape)

        torch.save(train_data_index, f'/home/zf/dataset/utkface/train.txt')
        torch.save(query_data_index, f'/home/zf/dataset/utkface/test.txt')
        torch.save(db_data_index, f'/home/zf/dataset/utkface/database.txt')

        data_index = {
            'train.txt': train_data_index,
            'test.txt': query_data_index,
            'database.txt': db_data_index
        }[fn]

    data = np.array(utkface.img_list)
    gender = np.array(utkface.gender_list)
    age = np.array(utkface.age_list)
    ethnicity = np.array(utkface.ethnicity_list)
    utkface.img_list = data[data_index]
    utkface.gender_list = gender[data_index]
    utkface.age_list = age[data_index]
    utkface.ethnicity_list = ethnicity[data_index]

    return utkface


def utk_multicls(**kwargs):
    nclass = 5
    transform = kwargs['transform']
    fn = kwargs['filename']
    reset = kwargs['reset']

    utkface = UTKface(nclass=5, ta='ethnicity', sa='age',
                      data_folder='/home/zf/dataset/utkface_multicls/utkface/UTKFace/', transform=transform,
                      multiclass=1)
    data = utkface.img_list
    gender = utkface.gender_list
    age = utkface.age_list
    ethnicity = utkface.ethnicity_list
    targets = ethnicity

    path = f'/home/zf/dataset/utkface_multicls/utkface/{fn}'

    load_data = fn == 'train.txt'
    load_data = load_data and (reset or not os.path.exists(path))

    if not load_data:
        print(f'Loading {path}')
        data_index = torch.load(path)
    else:
        train_data_index = []
        query_data_index = []
        db_data_index = []

        data_id = np.arange(data.shape[0])  # [0, 1, ...]

        for i in range(nclass):
            class_mask = targets == i
            index_of_class = data_id[class_mask].copy()  # index of the class [2, 10, 656,...]
            np.random.shuffle(index_of_class)

            query_n = 20  # query_n * nclass = query: 20 * 5 = 100

            index_for_query = index_of_class[:query_n].tolist()
            index_for_db = index_of_class[query_n:].tolist()
            index_for_train = index_for_db

            train_data_index.extend(index_for_train)
            query_data_index.extend(index_for_query)
            db_data_index.extend(index_for_db)

        train_data_index = np.array(train_data_index)
        query_data_index = np.array(query_data_index)
        db_data_index = np.array(db_data_index)

        print('train_data_index', train_data_index.shape)
        print('query_data_index', query_data_index.shape)
        print('db_data_index', db_data_index.shape)

        torch.save(train_data_index, f'/home/zf/dataset/utkface_multicls/utkface/train.txt')
        torch.save(query_data_index, f'/home/zf/dataset/utkface_multicls/utkface/test.txt')
        torch.save(db_data_index, f'/home/zf/dataset/utkface_multicls/utkface/database.txt')

        data_index = {
            'train.txt': train_data_index,
            'test.txt': query_data_index,
            'database.txt': db_data_index
        }[fn]

    data = np.array(utkface.img_list)
    gender = np.array(utkface.gender_list)
    age = np.array(utkface.age_list)
    ethnicity = np.array(utkface.ethnicity_list)
    utkface.img_list = data[data_index]
    utkface.gender_list = gender[data_index]
    utkface.age_list = age[data_index]
    utkface.ethnicity_list = ethnicity[data_index]

    return utkface


def utk_multicls2(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']
    reset = kwargs['reset']
    config = kwargs['config']
    transform_mode = kwargs['transform_mode']

    nclass = config['arch_kwargs']['nclass']
    n_labeled_samples = config['sample']['labeled_sample_num']
    ta = config['dataset_kwargs']['target_attribute']
    sa = config['dataset_kwargs']['sensitive_attribute']
    class_type = config['class_type']
    ds = config['dataset']

    utkface = UTKface(nclass=nclass, ta=ta, sa=sa, config=config, transform=transform)
    if ta == 'age':
        targets = utkface.age_list
    elif ta == 'gender':
        targets = utkface.gender_list
    elif ta == 'ethnicity':
        targets = utkface.ethnicity_list
    else:
        assert False, f'Invalid target attribute {ta}.'

    # obtain labeled samples
    data_index, generate_data = get_labeled_data(filename, reset, nclass, n_labeled_samples, targets, ds, ta, sa)

    data = np.array(utkface.img_list)
    gender = np.array(utkface.gender_list)
    age = np.array(utkface.age_list)
    ethnicity = np.array(utkface.ethnicity_list)

    utkface.img_list = data[data_index]
    utkface.gender_list = gender[data_index]
    utkface.age_list = age[data_index]
    utkface.ethnicity_list = ethnicity[data_index]

    original_img_files = utkface.img_list.copy()
    ################### diffusion model to generate labeled samples, with same target attribute and different sensitive attribute #######################
    # 是否使用增强的数据
    if config['aug_for_labeled_sample'] and transform_mode == 'train':
        # 首先查看relation.pt文件，里面记录的是标签样本与增强样本的文件名，看看在文件夹中是否对应，不对应的话，重新生成
        relation_path = f'./dataset/{ds}/{ds}_{ta}_{sa}_{nclass}/relations_{n_labeled_samples}.pt'
        sample_path = f'./dataset/generated_data/{ds}_{ta}_{sa}_{nclass}/samples_{n_labeled_samples}'
        relations = get_relations(relation_path, sample_path,
                                  original_img_files)  # 返回False说明train.txt中的样本与存在的生成样本不匹配，需要重新生成
        if config['sample']['generate_labeled_sample'] or generate_data or reset or not relations:
            # generate_data表示是否重新生成增强数据，返回True则重新生成增强数据，那么对应的图像也应该是重新生成的
            # save the generated samples to the ./generated_data folder
            print('Generating labeled samples using conditional diffusion model...')
            generate_samples_use_cdm(utkface.img_list, ta, sa, config)

        # 将增强之后的数据也添加到utkface数据集中
        generated_image_path = f'./dataset/generated_data/{ds}_{ta}_{sa}_{nclass}/samples_{n_labeled_samples}'
        assert os.path.exists(generated_image_path), f'The generated data path {generated_image_path} does not exist.'
        generated_img_list = []
        generated_gender_list = []
        generated_age_list = []
        generated_ethnicity_list = []
        for img_name in os.listdir(generated_image_path):
            generated_img_list.append(img_name)

            generated_gender_list.append(int(img_name.split('_')[1] == '0'))
            if class_type == 0:
                generated_age_list.append(int(img_name.split('_')[0]) < 35)
                generated_ethnicity_list.append(int(img_name.split('_')[2] == '0'))
            elif class_type == 1:
                generated_age_list.append(int(img_name.split('_')[0]) < 35)
                generated_ethnicity_list.append(int(img_name.split('_')[2]))
            elif class_type == 2:
                generated_ethnicity_list.append(int(img_name.split('_')[2] == '0'))
                if int(img_name.split('_')[0]) < 20:
                    generated_age_list.append(0)
                elif int(img_name.split('_')[0]) < 40:
                    generated_age_list.append(1)
                elif int(img_name.split('_')[0]) < 60:
                    generated_age_list.append(2)
                elif int(img_name.split('_')[0]) < 80:
                    generated_age_list.append(3)
                else:
                    generated_age_list.append(4)
        generated_img_list = np.array(generated_img_list)
        generated_gender_list = np.array(generated_gender_list)
        generated_age_list = np.array(generated_age_list)
        generated_ethnicity_list = np.array(generated_ethnicity_list)

        utkface.img_list = np.concatenate((utkface.img_list, generated_img_list))
        utkface.gender_list = np.concatenate((utkface.gender_list, generated_gender_list))
        utkface.age_list = np.concatenate((utkface.age_list, generated_age_list))
        utkface.ethnicity_list = np.concatenate((utkface.ethnicity_list, generated_ethnicity_list))

    ################### end of diffusion model ##########

    ################### annotate unlabeled samples use LLM ###################
    if config['annotation_unlabeled_data'] and transform_mode == 'train':
        # 首先查看relation.pt文件，里面记录的是标签样本与增强样本的文件名，看看在文件夹中是否对应，不对应的话，重新生成
        low_threshold = config['low_confidence_threshold']
        high_threshold = config['high_confidence_threshold']
        str_low_threshold = str(low_threshold).replace('0.', '')
        str_high_threshold = str(high_threshold).replace('0.', '')
        relation_path = f'./dataset/{ds}/{ds}_{ta}_{sa}_{nclass}/pseudo_relations_{n_labeled_samples}_{str_low_threshold}_{str_high_threshold}.pt'
        sample_path = f'./dataset/pseudo_data/{ds}_{ta}_{sa}_{nclass}/samples_{n_labeled_samples}_{str_low_threshold}_{str_high_threshold}'
        relations = get_relations(relation_path, sample_path,
                                  original_img_files)  # 返回False说明train.txt中的样本与存在的生成样本不匹配，需要重新生成
        if not relations:
            print('Annotating unlabeled samples using LLM...')
            # 获取全部未标记数据
            train_ids_path = f'./dataset/{ds}/{ds}_{ta}_{sa}_{nclass}/{filename}'
            database_ids_path = f'./dataset/{ds}/{ds}_{ta}_{sa}_{nclass}/database_{n_labeled_samples}.txt'
            unlabeled_data_files = get_unlabeled_files(train_ids_path, database_ids_path, config)
            test_unlabeled_data_files = random.sample(unlabeled_data_files, 300)  # 随机选取100个未标记数据进行测试
            # 采用多模态表征来找到low_confidence的样本
            low_confidence_files = get_low_confidence_files(unlabeled_data_files, config['low_confidence_threshold'],
                                                            config)

            # 采用LLM方法对low_confidence的样本进行处理，获取high_confidence的样本
            labeled_LLM = config['labeled_LLM']
            high_confidence_files = get_high_confidence_files_for_utkface(low_confidence_files, original_img_files, config,
                                                                          labeled_LLM, low_threshold, high_threshold)
        else:
            # 不需要重新标注，直接读取
            high_confidence_files = os.listdir(sample_path)
        # 更新utkface数据集
        if len(high_confidence_files) != 0:
            pseudo_img_list = []
            pseudo_gender_list = []
            pseudo_age_list = []
            pseudo_ethnicity_list = []
            for img_name in high_confidence_files:
                pseudo_img_list.append(img_name)

                pseudo_gender_list.append(int(img_name.split('_')[1] == '0'))
                if class_type == 0:
                    pseudo_age_list.append(int(img_name.split('_')[0]) < 35)
                    pseudo_ethnicity_list.append(int(img_name.split('_')[2] == '0'))
                elif class_type == 1:
                    pseudo_age_list.append(int(img_name.split('_')[0]) < 35)
                    pseudo_ethnicity_list.append(int(img_name.split('_')[2]))
                elif class_type == 2:
                    pseudo_ethnicity_list.append(int(img_name.split('_')[2] == '0'))
                    if int(img_name.split('_')[0]) < 20:
                        pseudo_age_list.append(0)
                    elif int(img_name.split('_')[0]) < 40:
                        pseudo_age_list.append(1)
                    elif int(img_name.split('_')[0]) < 60:
                        pseudo_age_list.append(2)
                    elif int(img_name.split('_')[0]) < 80:
                        pseudo_age_list.append(3)
                    else:
                        pseudo_age_list.append(4)
            pseudo_img_list = np.array(pseudo_img_list)
            pseudo_gender_list = np.array(pseudo_gender_list)
            pseudo_age_list = np.array(pseudo_age_list)
            pseudo_ethnicity_list = np.array(pseudo_ethnicity_list)

            utkface.img_list = np.concatenate((utkface.img_list, pseudo_img_list))
            utkface.gender_list = np.concatenate((utkface.gender_list, pseudo_gender_list))
            utkface.age_list = np.concatenate((utkface.age_list, pseudo_age_list))
            utkface.ethnicity_list = np.concatenate((utkface.ethnicity_list, pseudo_ethnicity_list))
    ################### end of annotate unlabeled samples use LLM ###################
    return utkface

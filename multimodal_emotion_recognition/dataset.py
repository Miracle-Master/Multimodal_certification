from datasets.ravdess import RAVDESS
import copy
def get_training_set(opt, spatial_transform=None, audio_transform=None):
    assert opt.dataset in ['RAVDESS'], print('Unsupported dataset: {}'.format(opt.dataset))
    opt = copy.deepcopy(opt)
    opt.phase = "train"
    if opt.dataset == 'RAVDESS':
        training_data = RAVDESS(
            opt,opt.annotation_path,
            'training',
            spatial_transform=spatial_transform, data_type='audiovisual', audio_transform=audio_transform)
    return training_data


def get_validation_set(opt, spatial_transform=None, audio_transform=None):
    assert opt.dataset in ['RAVDESS'], print('Unsupported dataset: {}'.format(opt.dataset))
    opt = copy.deepcopy(opt)
    opt.phase = "test"
    if opt.dataset == 'RAVDESS':
        validation_data = RAVDESS(
            opt, opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform, data_type = 'audiovisual', audio_transform=audio_transform)
    return validation_data


def get_test_set(opt, spatial_transform=None, audio_transform=None):
    assert opt.dataset in ['RAVDESS'], print('Unsupported dataset: {}'.format(opt.dataset))
    assert opt.test_subset in ['val', 'test']
    opt = copy.deepcopy(opt)
    opt.phase = "test"
    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'RAVDESS':
        test_data = RAVDESS(
            opt,opt.annotation_path,
            subset,
            spatial_transform=spatial_transform, data_type='audiovisual',audio_transform=audio_transform)
    return test_data

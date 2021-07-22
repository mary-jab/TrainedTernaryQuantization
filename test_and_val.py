
import torch
from model import device
import os
import data_preprocess
from model import get_model_to_quantify, device, get_model_full
import main_original
import numpy as np


def validation_loss(model, test_loader, loss_fn):
    model.eval()
    with torch.no_grad():
        total_img = 0
        total_loss = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            labels = torch.max(labels, 1)[1]

            outputs = model(images.float())
            total_img += labels.size(0)
            loss = loss_fn(outputs, labels)
            total_loss += loss
    model.train()

    return total_loss / total_img


# def test_on_dataloader(model, image_loader):
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for images, labels in image_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             labels = torch.max(labels, 1)[1]
#             model = model.double()
#             outputs = model(images.double())
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#         accuracy = 100 * correct / total
#
#     return len(image_loader.dataset.labels), accuracy


def test_on_dataloader(model, image_loader, arr=[]):
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0

        bt = image_loader.dataset.labels.shape[0]

        num_cls = image_loader.dataset.labels.shape[1]
        TP = np.zeros(num_cls)
        TN = np.zeros(num_cls)
        FP = np.zeros(num_cls)
        FN = np.zeros(num_cls)

        for images, labels in image_loader:
            images = images.to(device)
            labels = labels.to(device)
            labels = torch.max(labels, 1)[1]
            outputs = model(images.float())
            _, predicted = torch.max(outputs.data, 1)
            for cls in range(num_cls):
                prd_cls = predicted == cls
                grn_cls = labels == cls

                TP[cls] += ((prd_cls == True) & (grn_cls == True)).sum().item()
                TN[cls] += ((prd_cls == False) & (grn_cls == False)).sum().item()
                FP[cls] += ((prd_cls == True) & (grn_cls == False)).sum().item()
                FN[cls] += ((prd_cls == False) & (grn_cls == True)).sum().item()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * (precision * recall)/(precision + recall)
        arr.append(precision)
        arr.append(recall)
        arr.append(F1)
        accuracy = 100 * correct / total

    return len(image_loader.dataset.labels), accuracy


def calc_loss(model, image_loader, loss_fn):
    model.eval()
    with torch.no_grad():
        total_img = 0
        total_loss = 0
        for images, labels in image_loader:
            images = images.to(device)
            labels = labels.to(device)
            labels = torch.max(labels, 1)[1]

            outputs = model(images.float())
            total_img += labels.size(0)
            loss = loss_fn(outputs, labels)
            total_loss += loss
    # model.train()

    return total_loss / total_img.__float__()
    # total_loss = 0
    # total_num_img = 0
    # all_labels = None
    # all_outputs = None
    # with torch.no_grad():
    #     for images, labels in image_loader:
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         labels = torch.max(labels, 1)[1]
    #         outputs = model(images.float())
    #         if all_labels is None:
    #             all_labels = labels
    #             all_outputs = outputs
    #         else:
    #             all_labels = torch.cat((all_labels, labels), 0)
    #             all_outputs = torch.cat((all_outputs, outputs), 0)
    #
    #     total_loss = loss_fn(all_outputs, all_labels).item()
    #     total_num_img = all_labels.size(0)
    #     # total_num_img += labels.size(0)
    # return total_loss / total_num_img


def validation_acc(model, train_loader, test_loader,  max_acc,  epoch, matrices_acc_file_name, model_name=None):
    model.eval()
    train_data_sz, train_accuracy = test_on_dataloader(model, train_loader)
    test_data_sz, test_accuracy = test_on_dataloader(model, test_loader)

    string = f'\tepoch: {epoch}, ' \
             f'\ttrain sets: Accuracy on {train_data_sz} images: %{train_accuracy}, ' \
             f'\ttest sets: Accuracy on {test_data_sz} images: %{test_accuracy}\n'

    if max_acc < train_accuracy:
        print(string)

        max_acc = train_accuracy
        with open(matrices_acc_file_name, "a") as metrics_handle:
            metrics_handle.write(string)
        print('Saving model now!')
        save_model(model, model_name)

    file_name_sp = matrices_acc_file_name.split('.')
    with open(file_name_sp[0] + '_all.' + file_name_sp[1], "a") as metrics_handle:
        metrics_handle.write(string)

    model.train()
    return max_acc


def test(model, train_loader, test_loader):
    train_sz, train_acc = test_on_dataloader(model, train_loader)
    test_sz, test_acc = test_on_dataloader(model, test_loader)
    print(f'\ttrain sets: Accuracy on {train_sz} images: %{train_acc},'
    f'\ttest sets: Accuracy on {test_sz} images: %{test_acc}')


def save_model(model, model_name=None):
    if model_name is None:
        model_name = model.name
    dirname = os.path.dirname(__file__)
    dirname = os.path.join(dirname, 'weights')
    weightname = os.path.join(dirname, '{}.ckpt'.format(model_name))
    torch.save(model.state_dict(), weightname)


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.DoubleTensor')
    batch_size = 256
    # load_data
    dataset_name = 'uci_har'
    train_loader, test_loader = data_preprocess.load(batch_size=batch_size, dataset=dataset_name)
    ## load model
    model_type = 'float'
    kernel_size = 5
    idx = 0
    postfix = f'{dataset_name}_kernel_{kernel_size}_all_epoch300_{idx}'
    train_F_score = []
    test_F_score = []

    model = main_original.get_model(dataset_name, model_type=model_type, kernel_size=kernel_size)
    model = model.float()

    # load model with full precision trained weights
    dirname = os.path.dirname(__file__)
    dirname = os.path.join(dirname, 'weights')
    weightname = os.path.join(dirname, f'{postfix}_{model_type}.ckpt')
    model.load_state_dict(torch.load(weightname, map_location='cpu'))

    test_on_dataloader(model, train_loader, arr=train_F_score)
    test_on_dataloader(model, test_loader, arr=test_F_score)


    # Ternary
    model_type = 'ternary'
    idx = 0
    train_F_score = []
    test_F_score = []

    model = main_original.get_model(dataset_name, model_type=model_type, kernel_size=kernel_size)
    model = model.float()
    dirname = os.path.dirname(__file__)
    dirname = os.path.join(dirname, 'weights')
    weightname = os.path.join(dirname, f'{postfix}_{model_type}.ckpt')
    model.load_state_dict(torch.load(weightname, map_location='cpu'))
    test_on_dataloader(model, train_loader, arr=train_F_score)
    test_on_dataloader(model, test_loader, arr=test_F_score)
    i = 1
# motion_sense_drop25_all_epoch300_1_float
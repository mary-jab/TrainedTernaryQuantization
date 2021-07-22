from data_preprocess_motion_sense import load as load_motion_sense
from data_preprocess_uci_har import load as load_uci_har
import data_preprocess_uci_har

def load(batch_size=32, dataset='uci_har'):
    print(f"load {dataset} dataset")
    if dataset == 'uci_har':
        train_loader, test_loader = data_preprocess_uci_har.load(batch_size=batch_size)

        # train_loader, test_loader = load_uci_har(batch_size=batch_size)
    elif dataset == 'motion_sense':
        train_loader, test_loader = load_motion_sense(batch_size)
    else:
        raise Exception("Sorry, dataset could not be found!")

    return train_loader, test_loader

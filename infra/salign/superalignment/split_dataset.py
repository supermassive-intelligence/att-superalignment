import random


# Split a dataset into train and test
# Make sure to randomly shuffle the dataset
def split_dataset(dataset, min_test_samples=30):
    random.seed(42)
    random.shuffle(dataset)

    test_set_size = min_test_samples if min_test_samples > 0 else len(dataset)
    test_set_size_check = min_test_samples * 2

    if len(dataset) < test_set_size_check:
        test_set_size = len(dataset) // 2

    test_dataset = dataset[:test_set_size]
    train_dataset = dataset[test_set_size:] if test_set_size < len(dataset) else dataset

    return train_dataset, test_dataset

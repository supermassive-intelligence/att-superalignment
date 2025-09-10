import copy


def dedup(dataset):
    deduped_dataset = []
    seen = set()

    for item in dataset:
        if not item["question"] in seen:
            seen.add(item["question"])
            deduped_dataset.append(copy.deepcopy(item))

    return deduped_dataset

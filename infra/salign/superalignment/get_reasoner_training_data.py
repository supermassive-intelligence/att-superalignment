def get_reasoner_training_data(reasoners):
    training_data = []

    for reasoner in reasoners:
        training_data.extend(reasoner.get_training_data())

    return training_data

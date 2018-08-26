import data


def visualize_algorithms(trial_id, algo_name='enhanced'):
    devices = data.load_devices(trial_id, algo_name=algo_name)
    wrist = devices[0]
    transitive = devices[2]

    wrist_oxygen = wrist[['oxygen']].values.flatten()
    true_oxygen = transitive[['oxygen']].values.flatten()

    assert(wrist_oxygen.shape == true_oxygen.shape)

    for o1, o2 in zip(wrist_oxygen, true_oxygen):
        print(o1)
        print(o2)
        exit(0)



if __name__ == '__main__':
    trial_id = 18
    visualize_algorithms(trial_id)
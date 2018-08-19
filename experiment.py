import data


def run():
    data.list_trials()
    trial_id = 15
    devices = data.load_devices(trial_id)

    # Align Timestamps
    wrist, fingertip = data.normalize_timestamps(devices['wrist'], devices['reflective'])

    # Extract Features
    fe = data.FeatureExtractor(window_size=100)

    wrist_features = fe.extract_wrist_features(wrist)


if __name__ == '__main__':
    run()
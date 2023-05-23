import yaml


def read_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            return config
    except yaml.YAMLError as exc:
        print(exc)
        return {}
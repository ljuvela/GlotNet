from glotnet.config import Config
import os

def test_config_from_json():

    filepath = 'config/config.json'
    config = Config.from_json(filepath=filepath)



def test_config_to_json():

    config = Config()
    file_dir = '/tmp/json_dump/config.json'
    os.makedirs(file_dir, exist_ok=True)
    config.to_json(filepath=os.path.join(file_dir, 'config.json'))


if __name__ == "__main__":

    test_config_from_json()
    test_config_to_json()


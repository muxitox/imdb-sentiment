import yaml
import re

def load_configuration(path='./configuration.yaml'):
    with open(path, 'r') as f:
        return yaml.load(f)

def clean_document(document):
    words = re.sub(r'[^a-zA-Z]', ' ', document).lower()
    return words
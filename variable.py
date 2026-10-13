import yaml
import os

dashapp_path= os.getcwd()
CONFIG_PATH = os.path.abspath(os.path.join(dashapp_path, "app.yaml"))

try:
    with open(CONFIG_PATH, "r") as file:
        yaml_config = yaml.safe_load(file).get("env", {})
# Fallback to empty dict if file is missing (e.g., in production)
except FileNotFoundError:
    yaml_config = {}
print(yaml_config)
def get_env(key, default=None):
    # Check local YAML first, fallback to system environment variables
    return yaml_config.get(key, os.environ.get(key, default))

# 2. Extract the environment variables dictionary

# 3. Fetch your variables safely with default values


import yaml
import os
try:
    with open(r"C:\Users\fd01912\Documents\Ernest\Dashboard\app.yaml", "r") as file:
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


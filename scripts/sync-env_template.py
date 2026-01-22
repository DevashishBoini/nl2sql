## script to sync env template file with actual .env file
from pathlib import Path    
from dotenv import load_dotenv  

env_path = Path('.') / '.env'
template_path = Path('.') / '.env-template'

## copy exactly the .env file to .env-template, but replace values with '<YOUR_VALUE_HERE>'
def sync_env_template():
    with open(env_path, "r") as env_file:
        lines = env_file.readlines()

    with open(template_path, "w") as template_file:
        for line in lines:
            stripped = line.strip()

            # Preserve comments and blank lines
            if not stripped or stripped.startswith("#"):
                template_file.write(line)
                continue

            # Handle export VAR=value
            if stripped.startswith("export "):
                prefix, rest = stripped.split(" ", 1)
                key = rest.split("=", 1)[0]
                template_file.write(f"{prefix} {key}=<YOUR_VALUE_HERE>\n")
                continue

            # Handle VAR=value (with optional inline comment)
            if "=" in line:
                key, _, comment = line.partition("=")
                key = key.strip()
                inline_comment = ""

                if "#" in comment:
                    _, _, inline_comment = comment.partition("#")
                    inline_comment = " #" + inline_comment.rstrip()

                template_file.write(f"{key}=<YOUR_VALUE_HERE>{inline_comment}\n")



## execute the sync
if __name__ == "__main__":
    sync_env_template()
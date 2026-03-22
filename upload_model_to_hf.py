from huggingface_hub import HfApi

api = HfApi()

repo_id = "username/reponame"

# create repo if it doesn't exist
api.create_repo(repo_id, exist_ok=True)

# upload entire folder
api.upload_folder(
    folder_path="local_repository_folder",
    repo_id=repo_id,
)
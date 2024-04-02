import random
import time
import wandb
import zipfile
import os
import warnings


def config_and_condition_checking(args):
    
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")
            

def upload_py_code(dir2save, tracker=None):
    os.makedirs('~/tmp', exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    zipfilename = f'~/tmp/py_files_{timestamp}.zip'
    # Create a ZipFile Object
    with zipfile.ZipFile(zipfilename, 'w') as zipObj:
        # Iterate over all the files in the directory
        for folderName, subfolders, filenames in os.walk(dir2save):
            for filename in filenames:
                if filename.endswith('.py'):
                    # Create complete filepath of file in directory
                    filePath = os.path.join(folderName, filename)
                    # Add file to zip
                    zipObj.write(filePath)

    # Upload the zip file to wandb
    tracker.log_artifact(zipfilename)
    # remove the zip file
    os.remove(zipfilename)
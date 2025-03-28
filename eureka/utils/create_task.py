import yaml
import os
import logging

# # Load the YAML file
# task = 'Cartpole'
# suffix = 'GPT'

def create_task(root_dir, task, env_name, suffix):
    # Special case for Crafter environments
    if env_name.startswith('crafter'):
        create_crafter_task(task, env_name, suffix)
        return
        
    # Create task YAML file 
    input_file = f"{root_dir}/cfg/task/{task}.yaml"
    output_file = f"{root_dir}/cfg/task/{task}{suffix}.yaml"
    with open(input_file, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)

    # Modify the "name" field
    data['name'] = f'{task}{suffix}'
    data['env']['env_name'] = f'{env_name}{suffix}'
    
    # Write the new YAML file
    with open(output_file, 'w') as new_yamlfile:
        yaml.safe_dump(data, new_yamlfile)

    # Create training YAML file
    input_file = f"{root_dir}/cfg/train/{task}PPO.yaml"
    output_file = f"{root_dir}/cfg/train/{task}{suffix}PPO.yaml"

    with open(input_file, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)

    # Modify the "name" field
    data['params']['config']['name'] = data['params']['config']['name'].replace(task, f'{task}{suffix}')

    # Write the new YAML file
    with open(output_file, 'w') as new_yamlfile:
        yaml.safe_dump(data, new_yamlfile)

def create_crafter_task(task, env_name, suffix):
    """Creates minimal configuration for Crafter environments"""
    # Create output directories for logs and results
    os.makedirs("logs", exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    
    # Log the task creation
    logging.info(f"Creating Crafter task: {task}{suffix}")
    
    # Create a simple task YAML that could be used for reference
    output_file = f"crafter_task_{suffix.lower()}.yaml"
    
    task_data = {
        'name': f'{task}{suffix}',
        'env': {
            'env_name': f'{env_name}{suffix}',
            'reward_type': 'gpt',
            'num_envs': 1,
            'episodes': 10,
        }
    }
    
    # Write the task YAML file
    with open(output_file, 'w') as yamlfile:
        yaml.safe_dump(task_data, yamlfile)
        
    logging.info(f"Created Crafter task configuration: {output_file}")
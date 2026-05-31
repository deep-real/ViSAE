import torch.multiprocessing as mp
from dictionary_learning.trainers import BatchTopKTrainer
from dictionary_learning.training import trainSAE
from dictionary_learning.utils import InfiniteDataLoader
import torch
from torch.utils.data import TensorDataset
import os
import logging
from datetime import datetime
from tqdm import tqdm
import time

def setup_logging(save_dir, layer, expansion_factor):
    """Setup logging for the training process"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create log file path
    log_file = os.path.join(save_dir, f'training_layer_{layer}_ef_{expansion_factor}.log')
    
    # Create a unique logger name to avoid conflicts
    logger_name = f'layer_{layer}_ef_{expansion_factor}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add file and console handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def train_layer_sae(layer, gpu_id, expansion_factor, task_id, total_tasks, use_cls_tokens=True):
    device = f"cuda:{gpu_id}"
    activation_dim = 768
    epochs = 50
    batch_size = 4096
    dtype = torch.float32
    
    # Calculate dictionary size based on expansion factor
    dict_size = activation_dim * expansion_factor
    
    # Determine token type and save path
    token_type = "CLS" if use_cls_tokens else "Image"
    token_slice = "[:, 0, :]" if use_cls_tokens else "[:, 1:, :]"
    
    # Create save directory based on token type
    save_dir = f'/Checkpoints/SAE/Layer-wise_MSCOCO/{token_type}/BatchTopK-orig_top-128/ef_{expansion_factor}/layer_{layer}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(save_dir, layer, expansion_factor)
    
    logger.info(f"[Task {task_id}/{total_tasks}] Starting training for Layer {layer} on {device}")
    logger.info(f"Token type: {token_type} tokens ({token_slice})")
    logger.info(f"Expansion factor: {expansion_factor} (dict_size={dict_size})")
    logger.info(f"Save directory: {save_dir}")
    logger.info(f"Training parameters: epochs={epochs}, batch_size={batch_size}, lr=3e-4")

    try:
        # Load and prepare data
        logger.info(f"[Task {task_id}/{total_tasks}] Loading activations from layer {layer}...")
        
        # Load data based on token type
        if use_cls_tokens:
            embeddings = torch.load(f'/activations/clip-b32_layer-{layer}_resid-post.pt')[:, 0, :]
        else:
            embeddings = torch.load(f'/activations/clip-b32_layer-{layer}_resid-post.pt')[:, 1:, :]
        
        embeddings = embeddings.reshape(-1, activation_dim)
        # embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # Normalize if needed
        logger.info(f"[Task {task_id}/{total_tasks}] Loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
        
        tensor_dataset = TensorDataset(embeddings)
        dataset = InfiniteDataLoader(tensor_dataset, batch_size=batch_size)
        steps = epochs * len(tensor_dataset) // batch_size
        logger.info(f"[Task {task_id}/{total_tasks}] Training steps: {steps}")
        
        trainer_cfgs = [{
            "trainer": BatchTopKTrainer,
            "steps": steps,
            "warmup_steps": 0,
            "decay_start": None,
            "activation_dim": activation_dim,
            "lr": 3e-4,
            "dict_size": dict_size,
            "k": 128,
            "device": device,
            "layer": str(layer),
            "lm_name": 'openai/clip-vit-base-patch32'
        }]
        
        logger.info(f"[Task {task_id}/{total_tasks}] Starting SAE training...")
        ae = trainSAE(
            data=dataset,
            trainer_configs=trainer_cfgs,
            steps=steps,
            log_steps=50,
            verbose=True,
            device=device,
            autocast_dtype=dtype,
            save_dir=save_dir,
        )
        
        logger.info(f"[Task {task_id}/{total_tasks}] ✅ Training completed successfully for Layer {layer}, ef={expansion_factor}, {token_type} tokens")
        
    except Exception as e:
        logger.error(f"[Task {task_id}/{total_tasks}] ❌ Training failed for Layer {layer}, ef={expansion_factor}, {token_type} tokens: {str(e)}")
        raise e
    
    finally:
        # Clean up logging handlers to avoid conflicts
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

def monitor_progress(processes, tasks, main_logger):
    """Monitor the progress of all training processes"""
    completed = 0
    total = len(processes)
    
    # Create progress bar
    pbar = tqdm(total=total, desc="Overall Progress", position=0)
    
    while completed < total:
        time.sleep(10)  # Check every 10 seconds
        
        # Count completed processes
        new_completed = sum(1 for p in processes if not p.is_alive())
        
        if new_completed > completed:
            # Update progress bar
            pbar.update(new_completed - completed)
            
            # Log completed tasks
            for i in range(completed, new_completed):
                task = tasks[i]
                token_type = "CLS" if task['use_cls'] else "Image"
                main_logger.info(f"✅ Completed: Layer {task['layer']}, ef={task['ef']}, {token_type} tokens (Task {i+1}/{total})")
            
            completed = new_completed
    
    pbar.close()
    main_logger.info("🎉 All training processes completed!")

if __name__ == '__main__':
    # Configuration
    USE_CLS_TOKENS = False  # Set to False for image patch tokens
    # expansion_factors = [2, 4, 8, 16, 32]
    expansion_factors = [8]
    
    # Setup main logging
    token_type = "CLS" if USE_CLS_TOKENS else "Image"
    main_log_dir = f'/Checkpoints/SAE/Layer-wise/{token_type}/BatchTopK-orig_top-128/logs'
    os.makedirs(main_log_dir, exist_ok=True)
    
    main_log_file = os.path.join(main_log_dir, f'main_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Setup main logger
    main_logger = logging.getLogger('main')
    main_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in main_logger.handlers[:]:
        main_logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(main_log_file)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    main_logger.addHandler(file_handler)
    main_logger.addHandler(console_handler)
    
    # Generate all tasks
    tasks = []
    for layer in range(12):
        for ef in expansion_factors:
            tasks.append({
                'layer': layer, 
                'ef': ef, 
                'use_cls': USE_CLS_TOKENS
            })
    
    total_tasks = len(tasks)
    
    main_logger.info("="*80)
    main_logger.info(f"🚀 STARTING SAE TRAINING PIPELINE")
    main_logger.info(f"🎯 Token type: {token_type} tokens ({'[:, 0, :]' if USE_CLS_TOKENS else '[:, 1:, :]'})")
    main_logger.info(f"📊 Total tasks: {total_tasks}")
    main_logger.info(f"🔧 Expansion factors: {expansion_factors}")
    main_logger.info(f"📁 Main log file: {main_log_file}")
    main_logger.info("="*80)
    
    # Print task overview
    main_logger.info("📋 Task Overview:")
    for i, task in enumerate(tasks):
        main_logger.info(f"  Task {i+1:2d}: Layer {task['layer']:2d}, ef={task['ef']:2d}, {token_type} tokens")
    
    main_logger.info("="*80)
    
    # Start all processes
    processes = []
    
    for i, task in enumerate(tasks):
        gpu_id = i % 8  # Distribute across 8 GPUs
        task_id = i + 1
        
        main_logger.info(f"🎯 Starting Task {task_id}/{total_tasks}: Layer {task['layer']}, ef={task['ef']}, {token_type} tokens on GPU {gpu_id}")
        
        p = mp.Process(
            target=train_layer_sae, 
            args=(task['layer'], gpu_id, task['ef'], task_id, total_tasks, USE_CLS_TOKENS)
        )
        p.start()
        processes.append(p)
        
        # Small delay to avoid overwhelming the system
        time.sleep(1)
    
    main_logger.info(f"✅ Started all {len(processes)} training processes")
    main_logger.info("="*80)
    
    # Monitor progress in a separate thread/process
    try:
        monitor_progress(processes, tasks, main_logger)
    except KeyboardInterrupt:
        main_logger.info("🛑 Received interrupt signal, terminating processes...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        main_logger.info("❌ Training pipeline interrupted")
    
    # Wait for all processes to complete
    for i, p in enumerate(processes):
        p.join()
    
    main_logger.info("="*80)
    main_logger.info("🎉 TRAINING PIPELINE COMPLETED!")
    main_logger.info("="*80)
"""
RF Inversion Pipeline
=====================
A pipeline script to run the Random Forest inversion workflow in sequence:
1. generate_synthetic_data.py - Generate synthetic training data
2. train_rf.py - Train the Random Forest classifier
3. test_rf.py - Test the trained model on validation data
4. deploy_rf.py - Deploy the model on field data

Author: Emad Ghaleh Noei
Email: emadghalenoei@gmail.com
Date: 2025 Nov 11
"""

import os
import sys
import logging
import subprocess
import time
from datetime import datetime

def setup_logging():
    """Set up logging for the pipeline."""
    log_dir = 'pipeline_logs'
    os.makedirs(log_dir, exist_ok=True)

    log_filename = f"rf_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_path

def run_script(script_name, description):
    """
    Run a Python script and handle execution.

    Parameters:
    -----------
    script_name : str
        Name of the Python script to run
    description : str
        Description of what the script does (for logging)

    Returns:
    --------
    bool : True if successful, False if failed
    """
    logging.info(f"🚀 Starting {description} ({script_name})...")
    start_time = time.time()

    try:
        # Run the script and capture output
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True
        )

        # Log script output
        if result.stdout:
            logging.info(f"📝 {script_name} output:\n{result.stdout}")
        if result.stderr:
            logging.warning(f"⚠️  {script_name} warnings/errors:\n{result.stderr}")

        execution_time = time.time() - start_time
        logging.info(f"✅ {description} completed successfully in {execution_time:.2f} seconds")
        return True

    except subprocess.CalledProcessError as e:
        execution_time = time.time() - start_time
        logging.error(f"❌ {description} failed after {execution_time:.2f} seconds")
        logging.error(f"Error output:\n{e.stderr}")
        logging.error(f"Return code: {e.returncode}")
        return False
    except FileNotFoundError:
        logging.error(f"❌ Script {script_name} not found")
        return False
    except Exception as e:
        logging.error(f"❌ Unexpected error running {script_name}: {str(e)}")
        return False

def check_prerequisites():
    """Check if required files and directories exist."""
    logging.info("🔍 Checking prerequisites...")

    required_dirs = [
        'field_data'
    ]

    # Check for required directories
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)

    if missing_dirs:
        logging.warning(f"⚠️  Missing directories: {missing_dirs}")
        logging.info("Creating missing directories...")
        for dir_path in missing_dirs:
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"📁 Created directory: {dir_path}")

    # Check for field data files (warn if missing, but don't fail)
    field_data_files = [
        'field_data/GRV_2D_Data.txt',
        'field_data/RTP_2D_Data.txt'
    ]

    missing_field_data = []
    for file_path in field_data_files:
        if not os.path.exists(file_path):
            missing_field_data.append(file_path)

    if missing_field_data:
        logging.warning(f"⚠️  Missing field data files: {missing_field_data}")
        logging.warning("The pipeline will create a default config, but you should add your field data files.")

    # Check for required Python scripts
    required_scripts = [
        'generate_synthetic_data.py',
        'train_rf.py',
        'test_rf.py',
        'deploy_rf.py'
    ]

    missing_scripts = []
    for script in required_scripts:
        if not os.path.exists(script):
            missing_scripts.append(script)

    if missing_scripts:
        logging.error(f"❌ Missing required scripts: {missing_scripts}")
        return False

    logging.info("✅ Basic prerequisites checked")
    return True

def check_synthetic_data_generated():
    """Check if synthetic data already exists to avoid regeneration."""
    training_data_dir = 'training_data'
    if os.path.exists(training_data_dir):
        npy_files = [f for f in os.listdir(training_data_dir) if f.endswith('.npy')]
        if npy_files:
            logging.info(f"📁 Found {len(npy_files)} existing synthetic data files")
            return True
    return False

def check_config_exists():
    """Check if configuration file already exists."""
    config_path = 'constants/input_config.yaml'
    if os.path.exists(config_path):
        logging.info("⚙️  Found existing configuration file")
        return True
    return False

def main():
    """Main pipeline execution function."""
    # Setup logging
    log_path = setup_logging()
    logging.info("🎯 Starting RF Inversion Pipeline")
    logging.info(f"📄 Pipeline log: {log_path}")

    # Create constants directory if it doesn't exist
    os.makedirs('constants', exist_ok=True)

    # Check prerequisites
    if not check_prerequisites():
        logging.error("❌ Pipeline aborted due to missing prerequisites")
        sys.exit(1)

    # Check if synthetic data and config already exist
    synthetic_data_exists = check_synthetic_data_generated()
    config_exists = check_config_exists()

    # Define the pipeline steps
    pipeline_steps = [
        {
            'script': 'generate_synthetic_data.py',
            'description': 'Synthetic Data Generation and Configuration Setup',
            'critical': not synthetic_data_exists,  # Only critical if data doesn't exist
            'skip_if_exists': synthetic_data_exists,
            'note': 'Will create default config if none exists'
        },
        {
            'script': 'train_rf.py',
            'description': 'Random Forest Model Training',
            'critical': True
        },
        {
            'script': 'test_rf.py',
            'description': 'Model Testing and Validation',
            'critical': False
        },
        {
            'script': 'deploy_rf.py',
            'description': 'Model Deployment on Field Data',
            'critical': True
        }
    ]

    # Execute pipeline steps
    pipeline_start = time.time()
    success_count = 0
    failed_steps = []
    skipped_steps = []

    for step in pipeline_steps:
        # Skip step if data already exists and step is marked as skippable
        if step.get('skip_if_exists', False):
            logging.info(f"⏭️  Skipping {step['description']} - data already exists")
            if 'note' in step:
                logging.info(f"   📝 Note: {step['note']}")
            skipped_steps.append(step['script'])
            success_count += 1  # Count as success since we're intentionally skipping
            continue

        success = run_script(step['script'], step['description'])

        if success:
            success_count += 1
        else:
            failed_steps.append(step['script'])
            if step['critical']:
                logging.error(f"❌ Pipeline aborted due to critical failure in {step['script']}")
                break

    # Pipeline summary
    total_time = time.time() - pipeline_start
    logging.info("=" * 50)
    logging.info("📊 PIPELINE EXECUTION SUMMARY")
    logging.info(f"✅ Successful steps: {success_count}/{len(pipeline_steps)}")

    if skipped_steps:
        logging.info(f"⏭️  Skipped steps: {skipped_steps}")

    if failed_steps:
        logging.info(f"❌ Failed steps: {failed_steps}")
        logging.info("🚨 Pipeline completed with errors")
    else:
        logging.info("🎉 Pipeline completed successfully!")

    # Final recommendations
    logging.info("💡 RECOMMENDATIONS:")
    if not config_exists and os.path.exists('constants/input_config.yaml'):
        logging.info("   - Review and modify constants/input_config.yaml for your specific use case")

    if synthetic_data_exists:
        logging.info("   - Using existing synthetic data. Delete training_data/ folder to regenerate")

    logging.info(f"⏱️  Total execution time: {total_time:.2f} seconds")
    logging.info("=" * 50)

    # Exit with appropriate code
    if failed_steps:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
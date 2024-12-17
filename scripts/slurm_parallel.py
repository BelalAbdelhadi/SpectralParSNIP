import os
import math
import time

# List of datasets to process
datasets = ['sdss_data', 'sdss_blue', 'ps_data', 'ps_blue', 'csp_data', 'csp_blue', 'swift_data', 'swift_blue']

# Model path
ps_model_path = './model_7e-5.pt'

# Number of light curves to process per job
light_curves_per_job = 10

# Current working directory
pwd = os.getcwd()

# Count light curves for splitting
def count_light_curves(dataset):
    # Assuming lcdata.read_hdf5() is how we load the dataset
    import lcdata
    data = lcdata.read_hdf5(f'./data/{dataset}.h5')
    return len(data.light_curves)

# Loop over datasets to generate separate SLURM scripts for each dataset and chunk
for ds in datasets:

    #Split Dataset into chunks
    total_light_curves = count_light_curves(ds)
    total_jobs = math.ceil(total_light_curves / light_curves_per_job)  # Calculate the number of jobs needed

    for i in range(total_jobs):
        script_path = f"/tmp/tmp_{ds}_{i}.sh"
        # Open a temporary file for each job for this dataset
        with open(script_path, 'w') as f:

            # Write the SLURM script to the file
            f.write(f"""#!/bin/bash
#SBATCH --mail-type=END,FAIL,REQUEUE   
#SBATCH --mail-user=belal@hawaii.edu 
#SBATCH --job-name={ds}_part{i}_parsnip
#SBATCH --partition=shared
#SBATCH --time=3-00:00:00 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G  # Adjust memory requirement
#SBATCH --error={ds}_{i}-%A.err  # %A - job ID
#SBATCH --output={ds}_{i}-%A.out  # %A - job ID

#Information for debugging
echo "Starting job at $(date)"
echo "Dataset: {ds}, Chunk: {i}, Processing light curves {i * light_curves_per_job} to {(i + 1) * light_curves_per_job}"

module purge
module load lang/Anaconda3/
# Print out environment information before activation
echo "Before activating the environment:"
which python

# Activate the virtual environment
source activate shakinaz

# Log the Python executable path after activation
echo "Python executable after activating environment:"
which python
echo "Active conda environment: $(conda info --envs | grep '*' | awk '{{print $1}}')"

# Check installed libraries
pip list

cd {pwd}
# Start time tracking
start_time=$(date +%s)
/home/belal/.conda/envs/shakinaz/bin/python mcmc_fit_salt.py --dataset {ds} --start {i * light_curves_per_job} --end={min((i + 1) * light_curves_per_job, total_light_curves)}
 > ./logs/log_{ds}_{i}.txt

# End time tracking
end_time=$(date +%s)
execution_time=$((end_time - start_time))

echo "Finished job at $(date)"
echo "Job execution time (seconds): $execution_time"
""")
        
        # Close the file
        f.close()

        #Submit the job script
        os.system(f"sbatch {script_path}")

import ray
import A_runner
import os
os.environ['RAY_AIR_NEW_OUTPUT'] = '0' #without this, it doesnt work on Jewels

# Initialize Ray - connect to the existing cluster
ray.init(address='auto')
#ray.init(address='auto',include_dashboard=False, ignore_reinit_error=True,log_to_driver=False, _temp_dir = '/p/home/jusers/cipolina-kun1/juwels/ray_tmp')

# Call the function from A_runner.py
A_runner.run_shapley_runner()

# Shutdown Ray
ray.shutdown()

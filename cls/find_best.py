import pandas as pd

# Load the CSV file into a pandas DataFrame
path = "/mnt/home/fenglinghao/log/SpikingTransformerBenchmark/cls/neuron_model/spikf_semm_cifar--cifar10--PSNTorch--thres_1.0--tau_2.0--seed_42--epoch_400--20250418-091611/log.csv"
df = pd.read_csv(path)

# Replace 'column_name' with the actual name of the column you want to find the max value for
max_value_top1 = df['eval_top1'].max()
max_value_top5 = df['eval_top5'].max()

# Print the maximum value
print(f"max top 1: {max_value_top1}")
print(f"max top 5: {max_value_top5}")
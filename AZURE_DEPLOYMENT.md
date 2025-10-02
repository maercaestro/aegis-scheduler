# 🚀 Azure VM Deployment - Step-by-Step Guide

## Prerequisites ✅

Before starting, ensure you have:
- Azure account with active subscription
- Azure CLI installed on your local machine
- Git repository with all files ready

## Step 1: Install Azure CLI (if not already installed)

### macOS:
```bash
brew install azure-cli
```

### Windows:
Download from: https://aka.ms/installazurecliwindows

### Linux:
```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

## Step 2: Deploy VM from Your Local Machine

1. **Navigate to your project directory:**
   ```bash
   cd /Users/abuhuzaifahbidin/Documents/GitHub/aegis-scheduler
   ```

2. **Choose deployment script based on your subscription:**

   **For Azure Student Subscription (Recommended):**
   ```bash
   ./deploy-student-vm.sh
   ```

   **For Regular Azure Subscription:**
   ```bash
   ./deploy-vm.sh
   ```

   **For Minimal Resources (Fits any quota):**
   ```bash
   ./deploy-minimal-vm.sh
   ```

   The script will:
   - Login to Azure (if needed)
   - Check quota limits
   - Create appropriate resource group
   - Generate SSH keys
   - Create VM within your quota limits
   - Display connection details

## Step 3: Connect to Your VM

After deployment completes, you'll see connection details. Connect using:
```bash
ssh -i ~/.ssh/aegis_vm_key azureuser@<VM_PUBLIC_IP>
```

## Step 4: Transfer Files to VM

**Option A: Using git (Recommended):**
```bash
# On the VM
git clone https://github.com/maercaestro/aegis-scheduler.git
cd aegis-scheduler
```

**Option B: Using scp:**
```bash
# From your local machine
scp -i ~/.ssh/aegis_vm_key -r /Users/abuhuzaifahbidin/Documents/GitHub/aegis-scheduler azureuser@<VM_IP>:~/
```

## Step 5: Setup VM Environment

On the VM, run the setup script:
```bash
cd aegis-scheduler
./setup-vm.sh
```

This will:
- Install Python 3.12 and dependencies
- Create virtual environment
- Install all required packages
- Install solvers (GLPK, CBC, SCIP)
- Verify installation

## Step 6: Activate Environment and Test

```bash
source activate-aegis.sh
python main.py --check-solvers
```

## Step 7: Run Long-Running Optimization

### For Interactive Run (shorter tests):
```bash
python main.py --vessel-count 6 --time-limit 7200  # 2 hours
```

### For Background Run (long optimization):
```bash
nohup python main.py --vessel-count 6 --time-limit 28800 > logs/optimization.log 2>&1 &
```

### Monitor Progress:
```bash
tail -f logs/optimization.log
```

### Check Background Jobs:
```bash
ps aux | grep python  # See running processes
jobs                   # See background jobs
```

## Step 8: Retrieve Results

Results will be saved in the `results/` directory. To download them:

```bash
# From your local machine
scp -i ~/.ssh/aegis_vm_key -r azureuser@<VM_IP>:~/aegis-scheduler/results ./
```

## VM Specifications 💻

### For Regular Azure Subscription:
- **Size**: Standard_D4s_v3
- **vCPUs**: 4
- **RAM**: 16GB
- **Storage**: Premium SSD
- **OS**: Ubuntu 22.04 LTS
- **Cost**: ~$140/month (pay only when running)

### For Azure Student Subscription:
- **Size**: Standard_B4ms or Standard_B2s
- **vCPUs**: 4 or 2 (depending on quota)
- **RAM**: 16GB or 4GB
- **Storage**: Standard SSD
- **OS**: Ubuntu 22.04 LTS
- **Cost**: Uses student credits (~$10-20/month equivalent)

## Optimization Recommendations 🎯

### For 24+ Hour Runs:
```bash
# Maximum performance
python main.py --vessel-count 8 --optimization-type throughput --time-limit 86400 --threads 8 --mip-gap 0.01

# Background with logging
nohup python main.py --vessel-count 8 --time-limit 86400 --threads 8 > logs/optimization_24h.log 2>&1 &
```

### Cost Optimization:
- **Start VM only when needed**
- **Stop VM when optimization completes**
- **Use Azure auto-shutdown policies**

### Monitoring Commands:
```bash
# Check system resources
htop
free -h
df -h

# Monitor optimization progress
tail -f logs/optimization.log
grep -i "objective\|progress\|solution" logs/optimization.log
```

## Troubleshooting 🔧

### If solver issues:
```bash
python main.py --check-solvers
pip install --upgrade pyomo
```

### If memory issues:
```bash
# Check memory usage
free -h
# Reduce vessel count or increase VM size
python main.py --vessel-count 4 --time-limit 28800
```

### If connection lost:
```bash
# Reconnect to VM
ssh -i ~/.ssh/aegis_vm_key azureuser@<VM_IP>
# Check if optimization is still running
ps aux | grep python
```

## Cost Management 💰

### Stop VM when not in use:
```bash
# From local machine
az vm stop --resource-group aegis-optimization-rg --name aegis-vm
```

### Start VM when needed:
```bash
az vm start --resource-group aegis-optimization-rg --name aegis-vm
```

### Delete resources when done:
```bash
az group delete --name aegis-optimization-rg --yes --no-wait
```

## Expected Timeline ⏱️

- **VM Creation**: 3-5 minutes
- **Environment Setup**: 5-10 minutes
- **Optimization Run**: 8-24+ hours depending on complexity
- **Total Setup Time**: ~15 minutes

---

## 🎉 You're Ready!

Your Azure VM is now configured for long-running optimization workloads with professional-grade performance and reliability!
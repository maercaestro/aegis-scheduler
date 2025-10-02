# 🚀 Azure Deployment from Aegis Environment - Complete Guide

## 📋 Prerequisites Checklist

Before we start, make sure you have:
- ✅ Azure Student account
- ✅ Your optimization working locally in `aegis` environment
- ✅ All files in `/Users/abuhuzaifahbidin/Documents/GitHub/aegis-scheduler`

## 🎯 Step-by-Step Deployment Process

### **Step 1: Activate Aegis Environment**
```bash
# Navigate to your project
cd /Users/abuhuzaifahbidin/Documents/GitHub/aegis-scheduler

# Activate aegis virtual environment
source aegis/bin/activate

# Verify environment
python main.py --check-solvers
```

### **Step 2: Install Azure CLI & Deploy**
```bash
# Run the automated setup script
./install-and-deploy.sh
```

This script will:
1. ✅ Check you're in aegis environment
2. ✅ Install Azure CLI (if needed)
3. ✅ Login to Azure
4. ✅ Show your subscriptions
5. ✅ Let you choose VM size
6. ✅ Deploy the VM

### **Step 3: Manual Azure CLI Installation (if script fails)**

If the automated installation fails, install Azure CLI manually:

**Option A: Using Homebrew (Recommended for macOS)**
```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Azure CLI
brew install azure-cli
```

**Option B: Direct Download**
```bash
curl -L https://aka.ms/InstallAzureCli | bash
```

### **Step 4: Login and Deploy**
```bash
# Login to Azure
az login

# Check your subscription
az account list --output table

# Deploy minimal VM (recommended for student accounts)
./deploy-minimal-vm.sh
```

### **Step 5: Connect to Your VM**

After successful deployment, you'll get connection details:
```bash
# Connect to VM (replace with your actual IP and key path)
ssh -i ~/.ssh/aegis_minimal_key azureuser@<VM_PUBLIC_IP>
```

### **Step 6: Setup VM Environment**
```bash
# On the VM
git clone https://github.com/maercaestro/aegis-scheduler.git
cd aegis-scheduler
./setup-vm.sh
```

### **Step 7: Run Optimization on VM**
```bash
# Activate environment on VM
source activate-aegis.sh

# Run optimization (adjusted for minimal VM)
python main.py --vessel-count 3 --time-limit 7200 --threads 2

# For background run
nohup python main.py --vessel-count 3 --time-limit 14400 > logs/optimization.log 2>&1 &
```

## 🎓 Student Subscription Optimizations

### VM Size Recommendations:
1. **Standard_B2s** (2 vCPUs, 4GB RAM) - Ultra budget-friendly
2. **Standard_B4ms** (4 vCPUs, 16GB RAM) - Better performance

### Optimization Settings for Student VMs:
```bash
# For 2 vCPU VM
python main.py --vessel-count 3 --time-limit 7200 --threads 2

# For 4 vCPU VM  
python main.py --vessel-count 4 --time-limit 14400 --threads 4
```

## 💰 Cost Management for Students

### Stop VM when not in use:
```bash
az vm stop --resource-group aegis-minimal-rg --name aegis-minimal-vm
```

### Check your credit usage:
```bash
az consumption usage list --top 10
```

### Delete resources when done:
```bash
az group delete --name aegis-minimal-rg --yes --no-wait
```

## 🔧 Troubleshooting Common Issues

### 1. Quota Exceeded Error
**Solution**: Use smaller VM size
```bash
./deploy-minimal-vm.sh  # This uses Standard_B2s (2 vCPUs)
```

### 2. Azure CLI Not Found
**Solution**: Install manually
```bash
brew install azure-cli
# OR
curl -L https://aka.ms/InstallAzureCli | bash
```

### 3. Connection Refused
**Solution**: Wait for VM to fully start
```bash
# Wait 2-3 minutes after deployment
az vm show -d -g aegis-minimal-rg -n aegis-minimal-vm --query powerState
```

### 4. Out of Memory on VM
**Solution**: Reduce optimization parameters
```bash
python main.py --vessel-count 2 --time-limit 3600 --threads 1
```

## 📊 Expected Performance

### Minimal VM (2 vCPUs, 4GB RAM):
- **Vessel Count**: 2-3 vessels
- **Time Limit**: 2-6 hours
- **Expected Results**: Basic optimization

### Standard VM (4 vCPUs, 16GB RAM):
- **Vessel Count**: 4-6 vessels  
- **Time Limit**: 4-12 hours
- **Expected Results**: Good optimization quality

## 🎉 Quick Command Reference

```bash
# Complete deployment from scratch
cd /Users/abuhuzaifahbidin/Documents/GitHub/aegis-scheduler
source aegis/bin/activate
./install-and-deploy.sh

# Connect to VM
ssh -i ~/.ssh/aegis_minimal_key azureuser@<VM_IP>

# Setup on VM
git clone https://github.com/maercaestro/aegis-scheduler.git
cd aegis-scheduler && ./setup-vm.sh

# Run optimization
source activate-aegis.sh
python main.py --vessel-count 3 --time-limit 7200

# Monitor progress
tail -f logs/optimization.log

# Stop VM to save credits
az vm stop --resource-group aegis-minimal-rg --name aegis-minimal-vm
```

---

## 🚀 Ready to Deploy!

Run this command to start the complete deployment process:
```bash
source aegis/bin/activate && ./install-and-deploy.sh
```
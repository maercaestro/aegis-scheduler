#!/bin/bash

# Complete VM setup script using git clone
# Run this script on the Azure VM after connecting via SSH

set -e

echo "🚀 Setting up Aegis Scheduler from GitHub..."
echo "============================================="

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential tools
echo "🔧 Installing system dependencies..."
sudo apt install -y python3 python3-pip python3-venv git htop curl wget build-essential

# Clone the repository
echo "📥 Cloning aegis-scheduler repository..."
cd ~
git clone https://github.com/maercaestro/aegis-scheduler.git
cd aegis-scheduler

# Create Python virtual environment
echo "🐍 Setting up Python environment..."
python3 -m venv aegis
source aegis/bin/activate

# Upgrade pip and install dependencies
echo "📚 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Create systemd service for long-running optimizations
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/aegis-optimizer.service > /dev/null << 'EOF'
[Unit]
Description=Aegis Scheduler Optimization Service
After=network.target

[Service]
Type=simple
User=azureuser
WorkingDirectory=/home/azureuser/aegis-scheduler
Environment=PATH=/home/azureuser/aegis-scheduler/aegis/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/azureuser/aegis-scheduler/aegis/bin/python main.py --vessel-count 3 --time-limit 7200 --threads 2
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Create PM2-style process manager scripts
echo "📋 Creating process management scripts..."

# Start optimization script
cat > start-optimization.sh << 'EOF'
#!/bin/bash
source aegis/bin/activate

# Default optimization for minimal VM
VESSEL_COUNT=${1:-3}
TIME_LIMIT=${2:-7200}
THREADS=${3:-2}

echo "🚀 Starting optimization with:"
echo "  Vessels: $VESSEL_COUNT"
echo "  Time limit: $TIME_LIMIT seconds"
echo "  Threads: $THREADS"

# Run in background with logging
nohup python main.py \
    --vessel-count $VESSEL_COUNT \
    --time-limit $TIME_LIMIT \
    --threads $THREADS \
    --output-dir results \
    > logs/optimization_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo $! > optimization.pid
echo "✅ Optimization started with PID: $(cat optimization.pid)"
echo "📊 Monitor with: tail -f logs/optimization_$(date +%Y%m%d_%H%M%S).log"
EOF

# Stop optimization script
cat > stop-optimization.sh << 'EOF'
#!/bin/bash
if [ -f optimization.pid ]; then
    PID=$(cat optimization.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "🛑 Stopping optimization (PID: $PID)..."
        kill $PID
        rm optimization.pid
        echo "✅ Optimization stopped"
    else
        echo "⚠️  Process not running"
        rm optimization.pid
    fi
else
    echo "⚠️  No PID file found"
fi
EOF

# Status check script
cat > status-optimization.sh << 'EOF'
#!/bin/bash
if [ -f optimization.pid ]; then
    PID=$(cat optimization.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "✅ Optimization running (PID: $PID)"
        echo "📊 CPU usage: $(ps -p $PID -o %cpu --no-headers)%"
        echo "📊 Memory usage: $(ps -p $PID -o %mem --no-headers)%"
        echo "⏱️  Runtime: $(ps -p $PID -o etime --no-headers)"
    else
        echo "❌ Process not running (stale PID file)"
        rm optimization.pid
    fi
else
    echo "❌ No optimization running"
fi

# Show recent logs
echo ""
echo "📋 Recent logs:"
if ls logs/optimization_*.log 1> /dev/null 2>&1; then
    tail -10 $(ls -t logs/optimization_*.log | head -1)
else
    echo "No log files found"
fi
EOF

# Make scripts executable
chmod +x start-optimization.sh stop-optimization.sh status-optimization.sh

# Create directories
mkdir -p logs results

# Create systemd management script
cat > manage-service.sh << 'EOF'
#!/bin/bash
case "$1" in
    start)
        echo "🚀 Starting aegis-optimizer service..."
        sudo systemctl start aegis-optimizer
        ;;
    stop)
        echo "🛑 Stopping aegis-optimizer service..."
        sudo systemctl stop aegis-optimizer
        ;;
    restart)
        echo "🔄 Restarting aegis-optimizer service..."
        sudo systemctl restart aegis-optimizer
        ;;
    status)
        sudo systemctl status aegis-optimizer
        ;;
    enable)
        echo "✅ Enabling aegis-optimizer service for auto-start..."
        sudo systemctl enable aegis-optimizer
        ;;
    disable)
        echo "❌ Disabling aegis-optimizer service auto-start..."
        sudo systemctl disable aegis-optimizer
        ;;
    logs)
        echo "📋 Service logs:"
        sudo journalctl -u aegis-optimizer -f
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|enable|disable|logs}"
        exit 1
        ;;
esac
EOF

chmod +x manage-service.sh

# Reload systemd
sudo systemctl daemon-reload

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "🔧 Process Management Options:"
echo ""
echo "1️⃣ Manual Process Management:"
echo "   ./start-optimization.sh [vessels] [time_limit] [threads]"
echo "   ./status-optimization.sh"
echo "   ./stop-optimization.sh"
echo ""
echo "2️⃣ Systemd Service Management:"
echo "   ./manage-service.sh start    # Start service"
echo "   ./manage-service.sh status   # Check status"
echo "   ./manage-service.sh logs     # View logs"
echo "   ./manage-service.sh stop     # Stop service"
echo ""
echo "💡 Examples:"
echo "   ./start-optimization.sh 3 7200 2     # Minimal VM settings"
echo "   ./start-optimization.sh 6 14400 2    # More vessels, longer time"
echo ""
echo "📊 System Resources:"
echo "   CPUs: $(nproc)"
echo "   Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "   Disk: $(df -h / | tail -1 | awk '{print $4}') available"
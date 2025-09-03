#!/bin/bash

# Business Intelligence Platform - AWS EC2 Deployment Script
# Run this script on your EC2 instance to deploy the platform

set -e

echo "🚀 Business Intelligence Platform - AWS EC2 Deployment"
echo "======================================================"

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "🐳 Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Install Python and pip if not present (for local development)
if ! command -v python3 &> /dev/null; then
    echo "🐍 Installing Python..."
    sudo apt-get install -y python3 python3-pip python3-venv
fi

# Install system dependencies
echo "📚 Installing system dependencies..."
sudo apt-get install -y curl git htop nginx certbot python3-certbot-nginx

# Create application directory
APP_DIR="/opt/bi-platform"
sudo mkdir -p $APP_DIR
sudo chown $USER:$USER $APP_DIR

# Copy application files (assuming they're in current directory)
echo "📁 Setting up application files..."
cp -r . $APP_DIR/
cd $APP_DIR

# Create data and logs directories
mkdir -p data logs .streamlit

# Set up environment file template
if [ ! -f .env ]; then
    echo "🔑 Creating environment file template..."
    cat > .env << 'EOL'
# Business Intelligence Platform Configuration
# Copy this file to .env and fill in your API keys

# LLM API Keys (at least one is required)
GROQ_API_KEY=your_groq_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Application Settings
THREAD_ID=bi-platform-aws
USER_AGENT=BI-Platform/1.0

# Telemetry (optional)
OTEL_EXPORTER_OTLP_ENDPOINT=
OTEL_EXPORTER_OTLP_HEADERS=

# Database (optional - uses SQLite by default)
DATABASE_URL=sqlite:///data/bi_platform.db
EOL
    echo "⚠️  Please edit .env file and add your API keys!"
fi

# Create Streamlit configuration
echo "⚙️ Configuring Streamlit..."
cat > .streamlit/config.toml << 'EOL'
[server]
headless = true
address = "0.0.0.0"
port = 5000
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B35"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
EOL

# Build Docker image
echo "🏗️ Building Docker image..."
docker build -t bi-platform .

# Set up nginx reverse proxy
echo "🌐 Setting up Nginx reverse proxy..."
sudo tee /etc/nginx/sites-available/bi-platform << EOL
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain

    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        proxy_read_timeout 86400;
    }

    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOL

# Enable the site
sudo ln -sf /etc/nginx/sites-available/bi-platform /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl restart nginx

# Create systemd service for the platform
echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/bi-platform.service << EOL
[Unit]
Description=Business Intelligence Platform
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$APP_DIR
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOL

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable bi-platform

# Create startup script
echo "📜 Creating startup script..."
cat > start.sh << 'EOL'
#!/bin/bash
echo "🚀 Starting Business Intelligence Platform..."
docker-compose up -d
echo "✅ Platform is starting up..."
echo "🌐 Access your platform at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):5000"
EOL

chmod +x start.sh

# Create monitoring script
cat > monitor.sh << 'EOL'
#!/bin/bash
echo "📊 Business Intelligence Platform Status"
echo "======================================="
echo "Docker Status:"
docker-compose ps
echo ""
echo "Resource Usage:"
docker stats --no-stream
echo ""
echo "Recent Logs:"
docker-compose logs --tail=50
EOL

chmod +x monitor.sh

# Final instructions
echo ""
echo "🎉 Deployment Complete!"
echo "======================"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys: nano .env"
echo "2. Start the platform: ./start.sh"
echo "3. Monitor status: ./monitor.sh"
echo "4. Access via: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo 'YOUR_EC2_IP'):5000"
echo ""
echo "For SSL/HTTPS setup (recommended for production):"
echo "1. Update domain in /etc/nginx/sites-available/bi-platform"
echo "2. Run: sudo certbot --nginx -d your-domain.com"
echo ""
echo "To start automatically on boot:"
echo "sudo systemctl start bi-platform"
echo ""
echo "Troubleshooting:"
echo "- Check logs: docker-compose logs -f"
echo "- Restart: docker-compose restart"
echo "- Check nginx: sudo nginx -t && sudo systemctl status nginx"
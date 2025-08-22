# 🚀 Deployment Guide

Comprehensive deployment guide for the Agricultural Dataset Combination project. This guide covers containerization, cloud deployment, and production setup.

## 📋 **Deployment Overview**

### **Deployment Options**

1. **Local Deployment**: Direct installation on your machine
2. **Container Deployment**: Docker-based deployment
3. **Cloud Deployment**: AWS, Google Cloud, Azure
4. **Kubernetes Deployment**: Scalable container orchestration

### **Deployment Goals**

- **Reproducibility**: Consistent environments across deployments
- **Scalability**: Handle varying workloads
- **Reliability**: Robust error handling and recovery
- **Security**: Secure configuration and access control
- **Monitoring**: Comprehensive logging and metrics

## 🐳 **Container Deployment**

### **Docker Configuration**

#### **Base Dockerfile**

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src; print('Health check passed')" || exit 1

# Default command
CMD ["python", "scripts/combine_datasets.py"]
```

#### **Multi-stage Dockerfile (Production)**

```dockerfile
# Dockerfile.prod
# Build stage
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PATH=/root/.local/bin:$PATH

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy source code
COPY . .

# Create data directories
RUN mkdir -p /app/data/{input,output,intermediate}

# Expose ports (if needed)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src; print('OK')" || exit 1

# Run as non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "scripts/combine_datasets.py"]
```

#### **Docker Compose**

```yaml
# docker-compose.yml
version: '3.8'

services:
  dataset-processor:
    build: .
    container_name: agri-dataset-processor
    volumes:
      - ./data:/app/data
      - ./config:/app/config
      - ./logs:/app/logs
    environment:
      - PYTHONPATH=/app
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import src; print('OK')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  tensorboard:
    build: .
    container_name: agri-tensorboard
    volumes:
      - ./logs:/app/logs
    ports:
      - "6006:6006"
    command: tensorboard --logdir logs --host 0.0.0.0 --port 6006
    restart: unless-stopped
    depends_on:
      - dataset-processor

  jupyter:
    build: .
    container_name: agri-jupyter
    volumes:
      - ./notebooks:/app/notebooks
      - ./data:/app/data
    ports:
      - "8888:8888"
    command: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
    restart: unless-stopped
    environment:
      - JUPYTER_TOKEN=your-secure-token

volumes:
  data:
  logs:

networks:
  default:
    name: agri-network
```

### **Docker Deployment Commands**

```bash
# Build the image
docker build -t agricultural-dataset-combination:latest .

# Run the container
docker run -d \
  --name agri-processor \
  -v /path/to/data:/app/data \
  -v /path/to/config:/app/config \
  agricultural-dataset-combination:latest

# Run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f dataset-processor

# Stop services
docker-compose down

# Update and restart
docker-compose pull
docker-compose up -d --build
```

## ☁️ **Cloud Deployment**

### **AWS Deployment**

#### **EC2 Deployment**

```bash
# Launch EC2 instance
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t3.large \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxxx \
  --subnet-id subnet-xxxxxxxxx \
  --user-data file://user-data.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=AgriDatasetProcessor}]'
```

#### **User Data Script**

```bash
#!/bin/bash
# user-data.sh

# Update system
yum update -y

# Install Python 3.9
yum install -y python39 python39-pip python39-devel

# Install system dependencies
yum install -y mesa-libGL mesa-libEGL libXext libXrender

# Create application directory
mkdir -p /opt/agricultural-dataset
cd /opt/agricultural-dataset

# Clone repository
git clone https://github.com/selfishout/agricultural-dataset-combination.git .

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create data directories
mkdir -p data/{input,output,intermediate,logs}

# Set up systemd service
cat > /etc/systemd/system/agri-dataset.service << EOF
[Unit]
Description=Agricultural Dataset Processor
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/opt/agricultural-dataset
Environment=PATH=/opt/agricultural-dataset/venv/bin
ExecStart=/opt/agricultural-dataset/venv/bin/python scripts/combine_datasets.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl enable agri-dataset
systemctl start agri-dataset
```

#### **ECS Deployment**

```yaml
# task-definition.json
{
  "family": "agricultural-dataset",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::123456789012:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "dataset-processor",
      "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/agricultural-dataset:latest",
      "essential": true,
      "portMappings": [],
      "environment": [
        {
          "name": "PYTHONPATH",
          "value": "/app"
        },
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/agricultural-dataset",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### **Lambda Deployment**

```python
# lambda_function.py
import json
import boto3
import os
from src.dataset_combiner import DatasetCombiner

def lambda_handler(event, context):
    """AWS Lambda handler for dataset processing."""
    
    try:
        # Load configuration from environment variables
        config = {
            'storage': {
                'output_dir': os.environ['OUTPUT_DIR'],
                'intermediate_dir': os.environ['INTERMEDIATE_DIR']
            },
            'processing': {
                'target_size': [512, 512],
                'image_format': 'PNG',
                'augmentation': False,
                'quality_check': True
            }
        }
        
        # Initialize processor
        combiner = DatasetCombiner(config)
        
        # Process datasets
        results = combiner.combine_all_datasets()
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Dataset processing completed successfully',
                'results': results
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
```

### **Google Cloud Deployment**

#### **Compute Engine Deployment**

```bash
# Create instance
gcloud compute instances create agri-dataset-processor \
  --zone=us-central1-a \
  --machine-type=e2-standard-2 \
  --image-family=debian-11 \
  --image-project=debian-cloud \
  --metadata-from-file startup-script=startup-script.sh \
  --tags=http-server,https-server

# Create startup script
cat > startup-script.sh << 'EOF'
#!/bin/bash
# Install Python and dependencies
apt-get update
apt-get install -y python3 python3-pip python3-venv git

# Clone repository
cd /opt
git clone https://github.com/selfishout/agricultural-dataset-combination.git
cd agricultural-dataset-combination

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create data directories
mkdir -p data/{input,output,intermediate,logs}

# Set up systemd service
cat > /etc/systemd/system/agri-dataset.service << 'SERVICEEOF'
[Unit]
Description=Agricultural Dataset Processor
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/agricultural-dataset-combination
Environment=PATH=/opt/agricultural-dataset-combination/venv/bin
ExecStart=/opt/agricultural-dataset-combination/venv/bin/python scripts/combine_datasets.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICEEOF

# Enable and start service
systemctl enable agri-dataset
systemctl start agri-dataset
EOF
```

#### **Cloud Run Deployment**

```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: agricultural-dataset-processor
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "0"
        autoscaling.knative.dev/maxScale: "10"
    spec:
      containerConcurrency: 1
      timeoutSeconds: 3600
      containers:
      - image: gcr.io/PROJECT_ID/agricultural-dataset:latest
        env:
        - name: PYTHONPATH
          value: "/app"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
          requests:
            cpu: "1"
            memory: "2Gi"
```

### **Azure Deployment**

#### **Azure Container Instances**

```bash
# Deploy to Azure Container Instances
az container create \
  --resource-group your-resource-group \
  --name agri-dataset-processor \
  --image your-registry.azurecr.io/agricultural-dataset:latest \
  --dns-name-label agri-dataset-processor \
  --ports 8000 \
  --environment-variables \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO \
  --memory 4 \
  --cpu 2 \
  --registry-login-server your-registry.azurecr.io \
  --registry-username your-username \
  --registry-password your-password
```

#### **Azure Kubernetes Service**

```yaml
# aks-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agricultural-dataset-processor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agricultural-dataset-processor
  template:
    metadata:
      labels:
        app: agricultural-dataset-processor
    spec:
      containers:
      - name: dataset-processor
        image: your-registry.azurecr.io/agricultural-dataset:latest
        ports:
        - containerPort: 8000
        env:
        - name: PYTHONPATH
          value: "/app"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: agricultural-dataset-service
spec:
  selector:
    app: agricultural-dataset-processor
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 🚀 **Production Deployment**

### **Environment Configuration**

#### **Production Config**

```yaml
# config/production.yaml
storage:
  output_dir: "/data/combined_datasets"
  intermediate_dir: "/data/intermediate"
  backup_original: true
  compression: true

processing:
  target_size: [1024, 1024]
  image_format: "PNG"
  augmentation: true
  quality_check: true
  batch_size: 16
  num_workers: 8

splits:
  train_ratio: 0.7
  val_ratio: 0.2
  test_ratio: 0.1

logging:
  level: "INFO"
  file: "/logs/agricultural_dataset.log"
  max_size: "100MB"
  backup_count: 5

monitoring:
  enabled: true
  metrics_port: 8000
  health_check_interval: 30
```

#### **Environment Variables**

```bash
# .env.production
PYTHONPATH=/app
LOG_LEVEL=INFO
DATASET_SOURCE_DIR=/data/source_datasets
DATASET_OUTPUT_DIR=/data/combined_datasets
INTERMEDIATE_DIR=/data/intermediate
BACKUP_ORIGINAL=true
TARGET_SIZE=1024,1024
IMAGE_FORMAT=PNG
AUGMENTATION=true
QUALITY_CHECK=true
BATCH_SIZE=16
NUM_WORKERS=8
TRAIN_RATIO=0.7
VAL_RATIO=0.2
TEST_RATIO=0.1
```

### **Monitoring and Logging**

#### **Structured Logging**

```python
# src/logging_config.py
import logging
import logging.handlers
import json
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    """Structured JSON logging formatter."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)

def setup_logging(config):
    """Setup structured logging."""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, config.get('logging', {}).get('level', 'INFO')))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(StructuredFormatter())
    logger.addHandler(console_handler)
    
    # File handler
    if 'file' in config.get('logging', {}):
        file_handler = logging.handlers.RotatingFileHandler(
            config['logging']['file'],
            maxBytes=int(config['logging'].get('max_size', '100MB').replace('MB', '000000')),
            backupCount=config['logging'].get('backup_count', 5)
        )
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)
    
    return logger
```

#### **Health Checks**

```python
# src/health_check.py
import psutil
import os
import time
from typing import Dict, Any

class HealthChecker:
    """System health monitoring."""
    
    def __init__(self):
        self.start_time = time.time()
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        return {
            'status': 'healthy',
            'timestamp': time.time(),
            'uptime': time.time() - self.start_time,
            'system': self._check_system_resources(),
            'storage': self._check_storage(),
            'memory': self._check_memory(),
            'cpu': self._check_cpu()
        }
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        return {
            'memory_percent': psutil.virtual_memory().percent,
            'cpu_percent': psutil.cpu_percent(interval=1),
            'disk_percent': psutil.disk_usage('/').percent
        }
    
    def _check_storage(self) -> Dict[str, Any]:
        """Check storage availability."""
        try:
            stat = os.statvfs('/data')
            free_space = stat.f_frsize * stat.f_bavail
            total_space = stat.f_frsize * stat.f_blocks
            
            return {
                'free_space_gb': free_space / (1024**3),
                'total_space_gb': total_space / (1024**3),
                'free_percent': (free_space / total_space) * 100
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        memory = psutil.virtual_memory()
        return {
            'available_gb': memory.available / (1024**3),
            'total_gb': memory.total / (1024**3),
            'percent_used': memory.percent
        }
    
    def _check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage."""
        return {
            'percent': psutil.cpu_percent(interval=1),
            'count': psutil.cpu_count(),
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
        }
```

### **Security Configuration**

#### **Security Headers**

```python
# src/security.py
import os
import hashlib
import secrets
from typing import List, Optional

class SecurityManager:
    """Security configuration and validation."""
    
    def __init__(self, config: dict):
        self.config = config
        self.allowed_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        self.max_file_size = 100 * 1024 * 1024  # 100MB
    
    def validate_file_path(self, file_path: str) -> bool:
        """Validate file path for security."""
        # Check for path traversal attacks
        if '..' in file_path or file_path.startswith('/'):
            return False
        
        # Check file extension
        if not any(file_path.lower().endswith(ext) for ext in self.allowed_extensions):
            return False
        
        return True
    
    def validate_file_content(self, file_path: str) -> bool:
        """Validate file content for security."""
        try:
            # Check file size
            if os.path.getsize(file_path) > self.max_file_size:
                return False
            
            # Check file header (basic validation)
            with open(file_path, 'rb') as f:
                header = f.read(8)
                
                # PNG header
                if header.startswith(b'\x89PNG\r\n\x1a\n'):
                    return True
                
                # JPEG header
                if header.startswith(b'\xff\xd8\xff'):
                    return True
                
                # TIFF header
                if header.startswith(b'II\x2a\x00') or header.startswith(b'MM\x00\x2a'):
                    return True
                
                return False
                
        except Exception:
            return False
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate secure random token."""
        return secrets.token_urlsafe(length)
    
    def hash_file(self, file_path: str) -> str:
        """Generate SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
```

## 📊 **Performance Optimization**

### **Resource Optimization**

#### **Memory Management**

```python
# src/memory_manager.py
import gc
import psutil
import threading
import time
from typing import Optional

class MemoryManager:
    """Memory usage optimization and monitoring."""
    
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self):
        """Start memory monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Memory monitoring loop."""
        while self.monitoring:
            memory_percent = psutil.virtual_memory().percent
            
            if memory_percent > self.max_memory_percent:
                self._cleanup_memory()
            
            time.sleep(5)
    
    def _cleanup_memory(self):
        """Clean up memory when usage is high."""
        # Force garbage collection
        gc.collect()
        
        # Log memory cleanup
        print(f"Memory cleanup performed at {psutil.virtual_memory().percent}% usage")
    
    def get_memory_info(self) -> dict:
        """Get current memory information."""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent
        }
    
    def optimize_batch_size(self, image_size: tuple, target_memory_gb: float = 2.0) -> int:
        """Calculate optimal batch size based on available memory."""
        image_memory = image_size[0] * image_size[1] * 3 * 4  # RGB float32
        available_memory = target_memory_gb * (1024**3)
        return max(1, int(available_memory / image_memory))
```

#### **Parallel Processing**

```python
# src/parallel_processor.py
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Callable, Any
import os

class ParallelProcessor:
    """Parallel processing utilities."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
    
    def process_parallel(self, items: List[Any], process_func: Callable, 
                        use_threads: bool = False) -> List[Any]:
        """Process items in parallel."""
        executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_func, items))
        
        return results
    
    def process_in_chunks(self, items: List[Any], process_func: Callable, 
                          chunk_size: int = 1000) -> List[Any]:
        """Process items in chunks to manage memory."""
        results = []
        
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            chunk_results = self.process_parallel(chunk, process_func)
            results.extend(chunk_results)
            
            # Force garbage collection between chunks
            import gc
            gc.collect()
        
        return results
    
    def get_optimal_worker_count(self) -> int:
        """Get optimal number of worker processes."""
        cpu_count = mp.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Conservative approach: use fewer workers than CPU cores
        # and consider available memory
        optimal_workers = min(
            cpu_count - 1,  # Leave one core for system
            int(memory_gb / 2),  # Assume 2GB per worker
            self.max_workers
        )
        
        return max(1, optimal_workers)
```

## 🔧 **Deployment Scripts**

### **Deployment Automation**

#### **Deploy Script**

```bash
#!/bin/bash
# deploy.sh

set -e

# Configuration
PROJECT_NAME="agricultural-dataset-combination"
DOCKER_IMAGE="agricultural-dataset:latest"
DOCKER_REGISTRY="your-registry.azurecr.io"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    docker build -t $DOCKER_IMAGE .
    
    if [ $? -eq 0 ]; then
        log_info "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Push to registry
push_image() {
    log_info "Pushing image to registry..."
    docker tag $DOCK_IMAGE $DOCKER_REGISTRY/$DOCKER_IMAGE
    docker push $DOCKER_REGISTRY/$DOCKER_IMAGE
    
    if [ $? -eq 0 ]; then
        log_info "Image pushed successfully"
    else
        log_error "Failed to push image"
        exit 1
    fi
}

# Deploy with Docker Compose
deploy_compose() {
    log_info "Deploying with Docker Compose..."
    
    # Stop existing services
    docker-compose down
    
    # Pull latest images
    docker-compose pull
    
    # Start services
    docker-compose up -d
    
    if [ $? -eq 0 ]; then
        log_info "Deployment completed successfully"
    else
        log_error "Deployment failed"
        exit 1
    fi
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    # Wait for services to start
    sleep 30
    
    # Check if services are running
    if docker-compose ps | grep -q "Up"; then
        log_info "Health check passed"
    else
        log_error "Health check failed"
        exit 1
    fi
}

# Main deployment
main() {
    log_info "Starting deployment of $PROJECT_NAME"
    
    check_prerequisites
    build_image
    push_image
    deploy_compose
    health_check
    
    log_info "Deployment completed successfully!"
}

# Run main function
main "$@"
```

#### **Rollback Script**

```bash
#!/bin/bash
# rollback.sh

set -e

# Configuration
PROJECT_NAME="agricultural-dataset-combination"
BACKUP_TAG="backup-$(date +%Y%m%d-%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Rollback function
rollback() {
    log_info "Starting rollback..."
    
    # Stop current services
    docker-compose down
    
    # Check if backup image exists
    if docker images | grep -q "backup"; then
        log_info "Found backup image, restoring..."
        
        # Restore from backup
        docker tag agricultural-dataset:backup agricultural-dataset:latest
        
        # Start services with backup image
        docker-compose up -d
        
        log_info "Rollback completed successfully"
    else
        log_error "No backup image found"
        exit 1
    fi
}

# Main rollback
main() {
    log_info "Starting rollback of $PROJECT_NAME"
    rollback
    log_info "Rollback completed successfully!"
}

# Run main function
main "$@"
```

---

## 🎯 **Deployment Summary**

### **What We've Covered**

✅ **Container Deployment**: Docker and Docker Compose  
✅ **Cloud Deployment**: AWS, Google Cloud, Azure  
✅ **Production Setup**: Configuration and security  
✅ **Monitoring**: Health checks and logging  
✅ **Performance**: Memory and parallel processing  
✅ **Automation**: Deployment and rollback scripts  

### **Next Steps**

1. **Choose deployment strategy** based on your needs
2. **Set up monitoring** and logging infrastructure
3. **Configure security** and access controls
4. **Test deployment** in staging environment
5. **Monitor performance** and optimize as needed

---

**This deployment guide provides comprehensive coverage of deployment options. Choose the approach that best fits your infrastructure and requirements.** 🚀

---

<div align="center">

**Need More Details?** Check our [Complete Documentation](README.md) or [Source Code](https://github.com/selfishout/agricultural-dataset-combination)

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/selfishout)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ali-torabi)

</div>

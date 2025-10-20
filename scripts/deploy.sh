#!/bin/bash

# Deployment script for anomaly detection system
# This script handles the complete deployment pipeline

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="anomaly-transaction"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"
VERSION="${VERSION:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOGS_DIR="$PROJECT_ROOT/logs"
MODELS_DIR="$PROJECT_ROOT/data/models"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Deployment script for anomaly detection system

USAGE:
    $0 [OPTIONS] COMMAND

COMMANDS:
    build           Build Docker images
    test            Run tests
    train           Train models
    evaluate        Evaluate models
    deploy          Deploy to production
    rollback        Rollback to previous version
    status          Show deployment status
    logs            Show application logs
    clean           Clean up old deployments

OPTIONS:
    -e, --env       Environment (dev/staging/production) [default: production]
    -v, --version   Version tag [default: latest]
    -r, --registry  Docker registry [default: localhost:5000]
    -h, --help      Show this help message

EXAMPLES:
    $0 build
    $0 -e staging deploy
    $0 -v v1.2.3 deploy
    $0 test
    $0 logs

EOF
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check if required directories exist
    mkdir -p "$LOGS_DIR"
    mkdir -p "$MODELS_DIR"
    
    log_success "Prerequisites check passed"
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build API image
    log "Building API image..."
    docker build -f docker/Dockerfile -t "$DOCKER_REGISTRY/$PROJECT_NAME-api:$VERSION" .
    
    # Build Streamlit image
    log "Building Streamlit image..."
    docker build -f docker/Dockerfile.streamlit -t "$DOCKER_REGISTRY/$PROJECT_NAME-streamlit:$VERSION" .
    
    # Push images to registry
    if [ "$DOCKER_REGISTRY" != "localhost:5000" ]; then
        log "Pushing images to registry..."
        docker push "$DOCKER_REGISTRY/$PROJECT_NAME-api:$VERSION"
        docker push "$DOCKER_REGISTRY/$PROJECT_NAME-streamlit:$VERSION"
    fi
    
    log_success "Docker images built successfully"
}

# Run tests
run_tests() {
    log "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    # Install dependencies
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    
    # Run unit tests
    log "Running unit tests..."
    python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term
    
    # Run integration tests
    log "Running integration tests..."
    python -m pytest tests/test_integration.py -v
    
    log_success "All tests passed"
}

# Train models
train_models() {
    log "Training models..."
    
    cd "$PROJECT_ROOT"
    
    # Generate data if not exists
    if [ ! -f "data/raw/train.csv" ]; then
        log "Generating training data..."
        python scripts/generate_data.py --output-dir data/raw --n-samples 10000
    fi
    
    # Train models
    log "Training isolation forest..."
    python scripts/train_model.py --model isolation_forest --config configs/model-config.yaml
    
    log "Training autoencoder..."
    python scripts/train_model.py --model autoencoder --config configs/model-config.yaml
    
    log "Training ensemble..."
    python scripts/train_model.py --model ensemble --config configs/model-config.yaml
    
    log_success "Models trained successfully"
}

# Evaluate models
evaluate_models() {
    log "Evaluating models..."
    
    cd "$PROJECT_ROOT"
    
    # Run evaluation
    python scripts/evaluate_model.py \
        --config configs/model-config.yaml \
        --data data/raw/test.csv \
        --models-dir data/models \
        --output-dir evaluation_results
    
    log_success "Model evaluation completed"
}

# Deploy to production
deploy() {
    log "Deploying to $ENVIRONMENT environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create environment-specific docker-compose file
    cat > docker-compose.$ENVIRONMENT.yml << EOF
version: '3.8'

services:
  api:
    image: $DOCKER_REGISTRY/$PROJECT_NAME-api:$VERSION
    container_name: ${PROJECT_NAME}-api
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=$ENVIRONMENT
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  streamlit:
    image: $DOCKER_REGISTRY/$PROJECT_NAME-streamlit:$VERSION
    container_name: ${PROJECT_NAME}-streamlit
    ports:
      - "8501:8501"
    environment:
      - ENVIRONMENT=$ENVIRONMENT
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    depends_on:
      - api

  nginx:
    image: nginx:alpine
    container_name: ${PROJECT_NAME}-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx.conf:/etc/nginx/nginx.conf
      - ./docker/ssl:/etc/nginx/ssl
    restart: unless-stopped
    depends_on:
      - api
      - streamlit
EOF
    
    # Stop existing containers
    log "Stopping existing containers..."
    docker-compose -f docker-compose.$ENVIRONMENT.yml down || true
    
    # Start new containers
    log "Starting new containers..."
    docker-compose -f docker-compose.$ENVIRONMENT.yml up -d
    
    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 30
    
    # Health check
    log "Performing health checks..."
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "API is healthy"
    else
        log_error "API health check failed"
        exit 1
    fi
    
    if curl -f http://localhost:8501 > /dev/null 2>&1; then
        log_success "Streamlit is healthy"
    else
        log_error "Streamlit health check failed"
        exit 1
    fi
    
    log_success "Deployment completed successfully"
}

# Rollback deployment
rollback() {
    log "Rolling back deployment..."
    
    cd "$PROJECT_ROOT"
    
    # Get previous version
    PREVIOUS_VERSION=$(docker images --format "table {{.Tag}}" | grep "$PROJECT_NAME-api" | head -2 | tail -1)
    
    if [ -z "$PREVIOUS_VERSION" ]; then
        log_error "No previous version found"
        exit 1
    fi
    
    log "Rolling back to version: $PREVIOUS_VERSION"
    
    # Update docker-compose file with previous version
    sed -i "s/$VERSION/$PREVIOUS_VERSION/g" docker-compose.$ENVIRONMENT.yml
    
    # Deploy previous version
    docker-compose -f docker-compose.$ENVIRONMENT.yml down
    docker-compose -f docker-compose.$ENVIRONMENT.yml up -d
    
    log_success "Rollback completed"
}

# Show deployment status
show_status() {
    log "Deployment status:"
    
    cd "$PROJECT_ROOT"
    
    echo "=== Docker Containers ==="
    docker ps --filter "name=$PROJECT_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo -e "\n=== Service Health ==="
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "API: ✓ Healthy"
    else
        echo "API: ✗ Unhealthy"
    fi
    
    if curl -f http://localhost:8501 > /dev/null 2>&1; then
        echo "Streamlit: ✓ Healthy"
    else
        echo "Streamlit: ✗ Unhealthy"
    fi
    
    echo -e "\n=== Model Status ==="
    if [ -f "$MODELS_DIR/isolation_forest.joblib" ]; then
        echo "Isolation Forest: ✓ Trained"
    else
        echo "Isolation Forest: ✗ Not trained"
    fi
    
    if [ -f "$MODELS_DIR/autoencoder.pt" ]; then
        echo "Autoencoder: ✓ Trained"
    else
        echo "Autoencoder: ✗ Not trained"
    fi
    
    if [ -f "$MODELS_DIR/ensemble.joblib" ]; then
        echo "Ensemble: ✓ Trained"
    else
        echo "Ensemble: ✗ Not trained"
    fi
}

# Show logs
show_logs() {
    log "Showing application logs..."
    
    cd "$PROJECT_ROOT"
    
    # Show container logs
    docker-compose -f docker-compose.$ENVIRONMENT.yml logs -f --tail=100
}

# Clean up old deployments
cleanup() {
    log "Cleaning up old deployments..."
    
    # Remove old Docker images
    docker image prune -f
    
    # Remove old containers
    docker container prune -f
    
    # Remove old volumes
    docker volume prune -f
    
    # Clean up old logs
    find "$LOGS_DIR" -name "*.log" -mtime +7 -delete
    
    log_success "Cleanup completed"
}

# Main script logic
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -r|--registry)
                DOCKER_REGISTRY="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            build|test|train|evaluate|deploy|rollback|status|logs|clean)
                COMMAND="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    if [ -z "$COMMAND" ]; then
        log_error "No command specified"
        show_help
        exit 1
    fi
    
    # Execute command
    case $COMMAND in
        build)
            check_prerequisites
            build_images
            ;;
        test)
            check_prerequisites
            run_tests
            ;;
        train)
            check_prerequisites
            train_models
            ;;
        evaluate)
            check_prerequisites
            evaluate_models
            ;;
        deploy)
            check_prerequisites
            deploy
            ;;
        rollback)
            rollback
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        clean)
            cleanup
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"

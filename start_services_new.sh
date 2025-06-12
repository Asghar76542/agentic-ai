#!/bin/bash

# Enhanced AgenticSeek Start Services Script
# Load environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    source .env
else
    echo "Warning: .env file not found. Using default values."
fi

# Set default values if not in .env
export BACKEND_PORT=${BACKEND_PORT:-8000}
export FRONTEND_PORT=${FRONTEND_PORT:-3000}
export SEARXNG_PORT=${SEARXNG_PORT:-8080}

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    if [ ! -z "$COMPOSE_CMD" ]; then
        $COMPOSE_CMD down --remove-orphans 2>/dev/null || true
    fi
}

# Set up trap for cleanup on script exit
trap cleanup EXIT

# Validate WORK_DIR
if [ -z "$WORK_DIR" ]; then
    log_error "WORK_DIR environment variable is not set. Please set it in your .env file."
    exit 1
fi

if [ ! -d "$WORK_DIR" ]; then
    log_error "WORK_DIR ($WORK_DIR) does not exist or is not a directory."
    exit 1
fi

# Check directory size and warn if too large
log_info "Checking WORK_DIR size: $WORK_DIR"
if [[ "$OSTYPE" == "darwin"* ]]; then
    dir_size_bytes=$(du -s "$WORK_DIR" 2>/dev/null | awk '{print $1}' | xargs -I {} expr {} \* 512)
else
    dir_size_bytes=$(du -s --bytes "$WORK_DIR" 2>/dev/null | awk '{print $1}')
fi

max_size_bytes=$((2 * 1024 * 1024 * 1024)) # 2GB

if [ "$dir_size_bytes" -gt "$max_size_bytes" ]; then
    log_warning "WORK_DIR ($WORK_DIR) contains more than 2GB of data."
    log_warning "This may cause slow Docker performance. Consider using a smaller directory."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Determine deployment mode
DEPLOYMENT_MODE="core"
if [ "$1" = "full" ] || [ "$1" = "backend" ]; then
    DEPLOYMENT_MODE="full"
    log_info "Starting full deployment with backend and all services..."
else
    log_info "Starting core deployment (frontend and search services only)"
    log_info "Use './start_services.sh full' to start backend as well"
fi

# Check Docker installation
if ! command_exists docker; then
    log_error "Docker is not installed. Please install Docker first."
    log_info "Installation instructions:"
    log_info "- Ubuntu/Debian: sudo apt install docker.io"
    log_info "- CentOS/RHEL: sudo yum install docker"
    log_info "- macOS/Windows: Install Docker Desktop from https://www.docker.com/get-started/"
    exit 1
fi

# Check if Docker daemon is running
log_info "Checking Docker daemon status..."
if ! docker info &> /dev/null; then
    log_error "Docker daemon is not running or inaccessible."
    if [ "$(uname)" = "Linux" ]; then
        log_info "Attempting to start Docker service..."
        if sudo systemctl start docker &> /dev/null; then
            log_success "Docker daemon started successfully."
            sleep 3 # Give Docker time to fully start
        else
            log_error "Failed to start Docker daemon. Possible solutions:"
            log_info "1. Run with sudo: sudo $0 $@"
            log_info "2. Check service status: sudo systemctl status docker"
            log_info "3. Add user to docker group: sudo usermod -aG docker $USER (then logout/login)"
            exit 1
        fi
    else
        log_error "Please start Docker manually:"
        log_info "- macOS/Windows: Open Docker Desktop"
        log_info "- Linux: Run 'sudo systemctl start docker'"
        exit 1
    fi
else
    log_success "Docker daemon is running."
fi

# Determine Docker Compose command
if command_exists docker-compose; then
    COMPOSE_CMD="docker-compose"
elif docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    log_error "Docker Compose is not installed. Please install it first."
    log_info "Installation options:"
    log_info "- Ubuntu/Debian: sudo apt install docker-compose"
    log_info "- Via pip: pip install docker-compose"
    log_info "- As Docker plugin: Install Docker Desktop"
    exit 1
fi

log_info "Using Docker Compose command: $COMPOSE_CMD"

# Validate docker-compose.yml
if [ ! -f "docker-compose.yml" ]; then
    log_error "docker-compose.yml not found in the current directory."
    exit 1
fi

# Check for port conflicts
check_port() {
    local port=$1
    local service_name=$2
    if command_exists lsof && lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_warning "Port $port is already in use (needed for $service_name)"
        log_info "Please stop the service using this port or change the port in .env"
        return 1
    fi
    return 0
}

log_info "Checking for port conflicts..."
check_port $FRONTEND_PORT "frontend" || true
check_port $BACKEND_PORT "backend" || true
check_port $SEARXNG_PORT "searxng" || true

# Generate SEARXNG secret key
log_info "Generating SEARXNG secret key..."
if command -v openssl &> /dev/null; then
    export SEARXNG_SECRET_KEY=$(openssl rand -hex 32)
elif command -v python3 &> /dev/null; then
    export SEARXNG_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
else
    log_error "Neither openssl nor python3 is available to generate a secret key."
    exit 1
fi

# Clean up existing containers (optional)
log_info "Cleaning up existing containers..."
$COMPOSE_CMD down --remove-orphans &>/dev/null || true

# Pull latest images
log_info "Pulling latest Docker images..."
$COMPOSE_CMD pull --quiet || log_warning "Failed to pull some images (continuing anyway)"

# Start services based on deployment mode
if [ "$DEPLOYMENT_MODE" = "full" ]; then
    log_info "Starting full deployment..."
    
    # Start backend first
    log_info "Starting backend service..."
    if ! $COMPOSE_CMD up -d backend; then
        log_error "Failed to start backend container."
        $COMPOSE_CMD logs backend
        exit 1
    fi
    
    # Wait for backend health check
    log_info "Waiting for backend to be healthy..."
    timeout=60
    counter=0
    while [ $counter -lt $timeout ]; do
        if [ "$(docker inspect -f '{{.State.Running}}' agentic-seek-backend 2>/dev/null)" = "true" ]; then
            if command_exists curl && curl -f "http://localhost:$BACKEND_PORT/health" &>/dev/null; then
                log_success "Backend is healthy!"
                break
            fi
        fi
        
        if [ $counter -eq $((timeout-1)) ]; then
            log_error "Backend failed to become healthy after $timeout seconds"
            log_info "Backend logs:"
            $COMPOSE_CMD logs backend
            exit 1
        fi
        
        sleep 1
        counter=$((counter + 1))
        if [ $((counter % 10)) -eq 0 ]; then
            log_info "Still waiting for backend... ($counter/$timeout seconds)"
        fi
    done
    
    # Start all services
    log_info "Starting all services..."
    if ! $COMPOSE_CMD --profile full up -d; then
        log_error "Failed to start all containers."
        log_info "Check logs with: $COMPOSE_CMD logs"
        exit 1
    fi
else
    log_info "Starting core services (frontend and search only)..."
    if ! $COMPOSE_CMD --profile core up -d; then
        log_error "Failed to start core containers."
        log_info "Check logs with: $COMPOSE_CMD logs"
        exit 1
    fi
fi

# Wait for services to be ready
log_info "Waiting for services to be ready..."
sleep 5

# Health checks
log_info "Performing health checks..."

# Check frontend
if command_exists curl && curl -f "http://localhost:$FRONTEND_PORT" &>/dev/null; then
    log_success "Frontend is accessible at http://localhost:$FRONTEND_PORT"
else
    log_warning "Frontend might not be ready yet at http://localhost:$FRONTEND_PORT"
fi

# Check searxng
if command_exists curl && curl -f "http://localhost:$SEARXNG_PORT" &>/dev/null; then
    log_success "SearXNG is accessible at http://localhost:$SEARXNG_PORT"
else
    log_warning "SearXNG might not be ready yet at http://localhost:$SEARXNG_PORT"
fi

# Check backend (only in full mode)
if [ "$DEPLOYMENT_MODE" = "full" ]; then
    if command_exists curl && curl -f "http://localhost:$BACKEND_PORT/health" &>/dev/null; then
        log_success "Backend is accessible at http://localhost:$BACKEND_PORT"
    else
        log_warning "Backend might not be ready yet at http://localhost:$BACKEND_PORT"
    fi
fi

# Display final status
echo
log_success "🚀 AgenticSeek deployment completed!"
echo
log_info "📊 Service Status:"
log_info "- Frontend:  http://localhost:$FRONTEND_PORT"
log_info "- SearXNG:   http://localhost:$SEARXNG_PORT"
if [ "$DEPLOYMENT_MODE" = "full" ]; then
    log_info "- Backend:   http://localhost:$BACKEND_PORT"
    log_info "- API Docs:  http://localhost:$BACKEND_PORT/docs"
fi
echo
log_info "📝 Useful commands:"
log_info "- View logs:     $COMPOSE_CMD logs -f"
log_info "- Stop services: $COMPOSE_CMD down"
log_info "- Restart:       $COMPOSE_CMD restart"
log_info "- Update:        $COMPOSE_CMD pull && $COMPOSE_CMD up -d"
echo
log_info "🔍 If you encounter issues:"
log_info "- Check logs: $COMPOSE_CMD logs [service_name]"
log_info "- Restart services: $COMPOSE_CMD restart"
log_info "- Full restart: $COMPOSE_CMD down && ./start_services.sh $1"

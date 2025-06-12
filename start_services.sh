#!/bin/bash

# ==============================================================================
# AgenticSeek Enhanced Service Starter
# ==============================================================================
# This script starts all AgenticSeek services with improved error handling,
# logging, health checks, and monitoring capabilities.
# ==============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="AgenticSeek"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$LOG_DIR/services.pid"
HEALTH_CHECK_TIMEOUT=300  # 5 minutes
CHECK_INTERVAL=5

# Ensure logs directory exists
mkdir -p "$LOG_DIR"

# Logging functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_DIR/startup.log"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}" | tee -a "$LOG_DIR/startup.log"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}" | tee -a "$LOG_DIR/startup.log"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}" | tee -a "$LOG_DIR/startup.log"
}

log_info() {
    echo -e "${CYAN}[$(date '+%Y-%m-%d %H:%M:%S')] ℹ️  $1${NC}" | tee -a "$LOG_DIR/startup.log"
}

# Header
print_header() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                            🚀 AgenticSeek Launcher 🚀                       ║"
    echo "║                     Enhanced Multi-Agent AI System                          ║"
    echo "║                          Starting All Services...                           ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# System checks
check_system_requirements() {
    log "🔍 Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check available memory
    available_mem=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [ "$available_mem" -lt 2048 ]; then
        log_warning "Available memory is less than 2GB ($available_mem MB). Performance may be affected."
    fi
    
    # Check disk space
    available_disk=$(df . | awk 'NR==2{printf "%.0f", $4/1024}')
    if [ "$available_disk" -lt 5120 ]; then
        log_warning "Available disk space is less than 5GB ($available_disk MB). May affect operations."
    fi
    
    log_success "System requirements check completed"
}

# Environment setup
setup_environment() {
    log "🔧 Setting up environment..."
    
    # Create required directories
    mkdir -p "$SCRIPT_DIR/screenshots"
    mkdir -p "$SCRIPT_DIR/data/analytics"
    mkdir -p "$SCRIPT_DIR/data/dashboard"
    mkdir -p "$SCRIPT_DIR/data/knowledge_graph"
    mkdir -p "$SCRIPT_DIR/data/vector_memory"
    mkdir -p "$SCRIPT_DIR/workspace"
    
    # Set permissions
    chmod 755 "$SCRIPT_DIR/screenshots"
    chmod 755 "$SCRIPT_DIR/data"
    chmod -R 755 "$SCRIPT_DIR/data/"*
    chmod 755 "$SCRIPT_DIR/workspace"
    
    # Generate SEARXNG_SECRET_KEY if not set
    if [ -z "${SEARXNG_SECRET_KEY:-}" ]; then
        export SEARXNG_SECRET_KEY=$(openssl rand -hex 32 2>/dev/null || head -c 32 /dev/urandom | base64 | tr -d '/+=' | head -c 32)
        log_info "Generated SEARXNG_SECRET_KEY"
    fi
    
    log_success "Environment setup completed"
}

# Clean previous containers
cleanup_previous() {
    log "🧹 Cleaning up previous containers..."
    
    # Stop and remove containers if they exist
    if docker ps -a --format "table {{.Names}}" | grep -E "(backend|frontend|redis|searxng)" &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
        docker container prune -f &> /dev/null || true
    fi
    
    # Clean up dangling images
    docker image prune -f &> /dev/null || true
    
    log_success "Cleanup completed"
}

# Build images
build_images() {
    log "🔨 Building Docker images..."
    
    # Build with no cache to ensure fresh build
    if ! docker-compose -f "$COMPOSE_FILE" build --no-cache; then
        log_error "Failed to build Docker images"
        exit 1
    fi
    
    log_success "Docker images built successfully"
}

# Start services
start_services() {
    log "🚀 Starting services..."
    
    # Start services in detached mode
    if ! docker-compose -f "$COMPOSE_FILE" up -d; then
        log_error "Failed to start services"
        exit 1
    fi
    
    # Save container IDs to PID file
    docker-compose -f "$COMPOSE_FILE" ps -q > "$PID_FILE"
    
    log_success "Services started successfully"
}

# Health check function
check_service_health() {
    local service=$1
    local url=$2
    local timeout=${3:-30}
    
    log "🏥 Checking health of $service..."
    
    local count=0
    while [ $count -lt $timeout ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            log_success "$service is healthy"
            return 0
        fi
        
        count=$((count + 1))
        sleep 1
        
        if [ $((count % 10)) -eq 0 ]; then
            log_info "Still waiting for $service... ($count/${timeout}s)"
        fi
    done
    
    log_error "$service health check failed after ${timeout}s"
    return 1
}

# Comprehensive health checks
perform_health_checks() {
    log "🔍 Performing comprehensive health checks..."
    
    local all_healthy=true
    
    # Wait a bit for services to initialize
    log "⏳ Waiting for services to initialize..."
    sleep 10
    
    # Check SearXNG
    if ! check_service_health "SearXNG" "http://localhost:8080" 60; then
        all_healthy=false
    fi
    
    # Check Backend
    if ! check_service_health "Backend API" "http://localhost:8000/health" 60; then
        all_healthy=false
    fi
    
    # Check Frontend
    if ! check_service_health "Frontend" "http://localhost:3000" 60; then
        all_healthy=false
    fi
    
    # Check Redis (different approach since it's not HTTP)
    log "🏥 Checking health of Redis..."
    if docker exec redis redis-cli ping | grep -q PONG; then
        log_success "Redis is healthy"
    else
        log_error "Redis health check failed"
        all_healthy=false
    fi
    
    if [ "$all_healthy" = true ]; then
        log_success "All services are healthy and ready!"
        return 0
    else
        log_error "Some services failed health checks"
        return 1
    fi
}

# Display service status
show_service_status() {
    log "📊 Service Status Summary:"
    echo
    
    # Docker containers status
    echo -e "${CYAN}Docker Containers:${NC}"
    docker-compose -f "$COMPOSE_FILE" ps
    echo
    
    # Service URLs
    echo -e "${GREEN}🌐 Service Access URLs:${NC}"
    echo -e "  ${BLUE}Frontend (UI):${NC}        http://localhost:3000"
    echo -e "  ${BLUE}Backend (API):${NC}       http://localhost:8000"
    echo -e "  ${BLUE}API Documentation:${NC}   http://localhost:8000/docs"
    echo -e "  ${BLUE}SearXNG (Search):${NC}    http://localhost:8080"
    echo -e "  ${BLUE}Health Check:${NC}        http://localhost:8000/health"
    echo
    
    # System resources
    echo -e "${YELLOW}📈 System Resources:${NC}"
    echo -e "  ${BLUE}Memory Usage:${NC}        $(free -h | awk 'NR==2{printf "Used: %s / Total: %s (%.1f%%)", $3, $2, $3/$2*100}')"
    echo -e "  ${BLUE}Disk Usage:${NC}          $(df -h . | awk 'NR==2{printf "Used: %s / Total: %s (%s)", $3, $2, $5}')"
    echo -e "  ${BLUE}CPU Load:${NC}            $(uptime | awk -F'load average:' '{print $2}')"
    echo
    
    # Container resource usage
    echo -e "${PURPLE}🐳 Container Resources:${NC}"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
    echo
}

# Monitor services
monitor_services() {
    log "👀 Starting service monitoring..."
    
    # Create monitoring script
    cat > "$LOG_DIR/monitor.sh" << 'EOF'
#!/bin/bash
while true; do
    if ! docker-compose ps | grep -q "Up"; then
        echo "[$(date)] WARNING: Some services may be down" >> logs/monitor.log
    fi
    sleep 30
done
EOF
    
    chmod +x "$LOG_DIR/monitor.sh"
    nohup "$LOG_DIR/monitor.sh" &
    echo $! > "$LOG_DIR/monitor.pid"
    
    log_success "Service monitoring started"
}

# Error handler
handle_error() {
    local exit_code=$?
    log_error "Script failed with exit code $exit_code"
    
    # Show container logs for debugging
    log "📋 Showing recent container logs for debugging:"
    docker-compose -f "$COMPOSE_FILE" logs --tail=20
    
    exit $exit_code
}

# Graceful shutdown handler
graceful_shutdown() {
    log "🛑 Received shutdown signal. Stopping services gracefully..."
    
    # Stop monitoring
    if [ -f "$LOG_DIR/monitor.pid" ]; then
        kill "$(cat "$LOG_DIR/monitor.pid")" 2>/dev/null || true
        rm -f "$LOG_DIR/monitor.pid"
    fi
    
    # Stop services
    docker-compose -f "$COMPOSE_FILE" down
    
    log_success "Services stopped gracefully"
    exit 0
}

# Main execution
main() {
    # Set up signal handlers
    trap handle_error ERR
    trap graceful_shutdown SIGINT SIGTERM
    
    # Start timing
    start_time=$(date +%s)
    
    print_header
    
    # Run all setup steps
    check_system_requirements
    setup_environment
    cleanup_previous
    build_images
    start_services
    
    # Health checks with retry
    local health_attempts=0
    local max_health_attempts=3
    
    while [ $health_attempts -lt $max_health_attempts ]; do
        if perform_health_checks; then
            break
        fi
        
        health_attempts=$((health_attempts + 1))
        if [ $health_attempts -lt $max_health_attempts ]; then
            log_warning "Health check attempt $health_attempts failed. Retrying in 10 seconds..."
            sleep 10
        fi
    done
    
    if [ $health_attempts -eq $max_health_attempts ]; then
        log_error "All health check attempts failed. Please check the logs."
        docker-compose -f "$COMPOSE_FILE" logs --tail=50
        exit 1
    fi
    
    # Start monitoring
    monitor_services
    
    # Show final status
    show_service_status
    
    # Calculate startup time
    end_time=$(date +%s)
    startup_time=$((end_time - start_time))
    
    # Success message
    echo
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                          🎉 SUCCESS! 🎉                                     ║${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}║                     AgenticSeek is now running!                             ║${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}║                 Startup completed in ${startup_time} seconds                              ║${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}║                Open http://localhost:3000 to get started                    ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo
    
    log_success "AgenticSeek startup completed successfully in ${startup_time} seconds"
    
    # Keep script running to maintain monitoring
    log "🔄 Keeping services running. Press Ctrl+C to stop all services."
    
    # Wait for signals
    while true; do
        sleep 10
        
        # Quick health check every minute
        if [ $(($(date +%s) % 60)) -eq 0 ]; then
            if ! docker-compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
                log_warning "Some services appear to be down. Check status with: docker-compose ps"
            fi
        fi
    done
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
# 🔧 AgenticSeek Auto-Fix Tools

## Quick Problem Resolution

Having issues with AgenticSeek setup? Try our automated fix tools:

### 🚀 One-Command Fix (Recommended)
```bash
./scripts/master_fix.sh
```
This runs all auto-fixes and resolves most common issues automatically.

### 🎯 Specific Issue Fixes

**ChromeDriver Problems:**
```bash
./scripts/fix_chromedriver.sh
```

**SearxNG Configuration Issues:**
```bash
./scripts/fix_searxng.sh
```

**MCP Server Problems:**
```bash
./scripts/fix_mcp_servers.sh
```

**General Configuration Validation:**
```bash
python3 scripts/validate_config.py
```

## What Gets Fixed

✅ **ChromeDriver Issues** - Automatic version detection and installation  
✅ **"No connection adapters found" errors** - HTTP prefix auto-fixing  
✅ **"SearxNG base URL must be provided" errors** - Environment setup  
✅ **MCP "spawn npx ENOENT" errors** - Node.js path configuration  
✅ **Configuration validation** - Comprehensive setup checking  

## After Running Fixes

1. Start services: `./start_services.sh`
2. Run AgenticSeek: `python3 cli.py`
3. Access web interface: `http://localhost:3000`

For detailed documentation, see: `docs/AUTO_FIX_TOOLS.md`

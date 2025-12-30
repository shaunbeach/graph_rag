#!/bin/bash
# Test script to verify RAG System installation
# Run this after installing to ensure everything works

set -e  # Exit on error

echo "======================================"
echo "RAG System Installation Test"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    ((TESTS_PASSED++))
}

fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    ((TESTS_FAILED++))
}

info() {
    echo -e "${YELLOW}ℹ INFO${NC}: $1"
}

# Test 1: Check if rag command exists
echo "Test 1: Checking if 'rag' command is available..."
if command -v rag &> /dev/null; then
    pass "rag command found"
else
    fail "rag command not found. Did you install the package?"
    exit 1
fi
echo ""

# Test 2: Check version
echo "Test 2: Checking version..."
if rag --version &> /dev/null; then
    VERSION=$(rag --version)
    pass "Version check: $VERSION"
else
    fail "Could not get version"
fi
echo ""

# Test 3: Check help
echo "Test 3: Checking help command..."
if rag --help &> /dev/null; then
    pass "Help command works"
else
    fail "Help command failed"
fi
echo ""

# Test 4: Create test workspace
echo "Test 4: Creating test workspace..."
TEST_WORKSPACE="/tmp/rag_test_$$"
mkdir -p "$TEST_WORKSPACE/docs"

if [ -d "$TEST_WORKSPACE" ]; then
    pass "Test workspace created at $TEST_WORKSPACE"
else
    fail "Could not create test workspace"
fi
echo ""

# Test 5: Create test documents
echo "Test 5: Creating test documents..."
cat > "$TEST_WORKSPACE/docs/test1.md" << 'EOF'
# Machine Learning Basics

Machine learning is a method of data analysis that automates analytical model building.
It is a branch of artificial intelligence based on the idea that systems can learn from data.
EOF

cat > "$TEST_WORKSPACE/docs/test2.md" << 'EOF'
# Python Programming

Python is a high-level, interpreted programming language.
It is widely used for web development, data analysis, and machine learning.
EOF

if [ -f "$TEST_WORKSPACE/docs/test1.md" ] && [ -f "$TEST_WORKSPACE/docs/test2.md" ]; then
    pass "Test documents created"
else
    fail "Could not create test documents"
fi
echo ""

# Test 6: Test ingestion
echo "Test 6: Testing document ingestion..."
info "This will download the embedding model (~400MB) on first run..."
if rag ingest "$TEST_WORKSPACE/docs/" --workspace "$TEST_WORKSPACE/workspace" > /dev/null 2>&1; then
    pass "Document ingestion successful"
else
    fail "Document ingestion failed"
fi
echo ""

# Test 7: Test status command
echo "Test 7: Testing status command..."
if rag status --workspace "$TEST_WORKSPACE/workspace" > /dev/null 2>&1; then
    pass "Status command works"
else
    fail "Status command failed"
fi
echo ""

# Test 8: Test history command
echo "Test 8: Testing history command..."
if rag history --workspace "$TEST_WORKSPACE/workspace" > /dev/null 2>&1; then
    pass "History command works"
else
    fail "History command failed"
fi
echo ""

# Test 9: Test query command
echo "Test 9: Testing query command..."
if rag query "what is machine learning?" --workspace "$TEST_WORKSPACE/workspace" > /dev/null 2>&1; then
    pass "Query command works"
else
    fail "Query command failed"
fi
echo ""

# Test 10: Test info command
echo "Test 10: Testing info command..."
if rag info > /dev/null 2>&1; then
    pass "Info command works"
else
    fail "Info command failed"
fi
echo ""

# Test 11: Check workspace structure
echo "Test 11: Verifying workspace structure..."
if [ -d "$TEST_WORKSPACE/workspace/vector_store" ]; then
    pass "Vector store directory created"
else
    fail "Vector store directory not found"
fi

if [ -f "$TEST_WORKSPACE/workspace/file_index.json" ]; then
    pass "File index created"
else
    info "File index not found (may be in vector_store directory)"
fi

if [ -f "$TEST_WORKSPACE/workspace/ingestion_log.json" ]; then
    pass "Ingestion log created"
else
    info "Ingestion log not found"
fi
echo ""

# Cleanup
echo "Cleaning up..."
read -p "Remove test workspace at $TEST_WORKSPACE? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$TEST_WORKSPACE"
    info "Test workspace removed"
else
    info "Test workspace kept at $TEST_WORKSPACE"
fi
echo ""

# Summary
echo "======================================"
echo "Test Summary"
echo "======================================"
echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed! ✓${NC}"
    echo ""
    echo "RAG System is ready to use!"
    echo ""
    echo "Next steps:"
    echo "  1. Read the Quick Start guide: docs/QUICK_START.md"
    echo "  2. Try the examples: examples/basic_usage.md"
    echo "  3. Start ingesting your documents!"
    exit 0
else
    echo -e "${RED}Some tests failed. ✗${NC}"
    echo ""
    echo "Please check:"
    echo "  1. Installation completed successfully"
    echo "  2. All dependencies installed"
    echo "  3. Python version >= 3.8"
    exit 1
fi

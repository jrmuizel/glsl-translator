# Test Documentation

This document describes the testing setup for the GLSL Type Checker project.

## Test Structure

### Test Organization

The tests are organized in `src/tests.rs` and cover the following areas:

#### 1. GLSLType Unit Tests
- **test_glsl_type_display**: Tests the Display implementation for various GLSL types
- **test_glsl_type_is_scalar**: Tests scalar type detection (bool, int, uint, float, double)
- **test_glsl_type_is_vector**: Tests vector type detection (vec2/3/4, ivec2/3/4, etc.)
- **test_glsl_type_is_matrix**: Tests matrix type detection (mat2/3/4)
- **test_glsl_type_is_numeric**: Tests numeric type detection (excludes bool types)
- **test_glsl_type_component_count**: Tests component counting for vectors and matrices
- **test_glsl_type_base_type**: Tests base type extraction for vectors and matrices
- **test_glsl_type_compatibility**: Tests implicit type conversion rules

#### 2. SymbolTable Unit Tests
- **test_symbol_table_scoping**: Tests variable scoping and shadowing
- **test_symbol_table_duplicate_declaration**: Tests duplicate variable detection
- **test_symbol_table_functions**: Tests built-in function availability and custom function declaration

#### 3. SimpleTypeChecker Integration Tests
- **test_type_checker_simple_function**: Tests basic function type checking
- **test_type_checker_arithmetic_operations**: Tests arithmetic expression type checking
- **test_type_checker_vector_operations**: Tests vector operations and function calls
- **test_type_checker_function_definition**: Tests function definition and calls
- **test_type_checker_type_errors**: Tests error detection and reporting
- **test_type_checker_new**: Tests type checker initialization

#### 4. Error Handling Tests
- **test_type_error_display**: Tests error message formatting
- **test_invalid_glsl_parsing**: Tests invalid GLSL code handling
- **test_empty_translation_unit**: Tests empty input handling
- **test_multiple_variable_declarations**: Tests complex variable declarations

#### 5. Built-in Function Tests
- **test_builtin_functions_available**: Tests that all expected built-in functions are registered

## Running Tests

### Local Testing
```bash
# Run all tests
cargo test

# Run tests with verbose output
cargo test -- --nocapture

# Run specific test
cargo test test_glsl_type_is_numeric

# Run tests in release mode
cargo test --release
```

### Code Quality Checks
```bash
# Format code
cargo fmt

# Check formatting
cargo fmt --check

# Run clippy linter
cargo clippy --all-targets --all-features -- -D warnings

# Build documentation
cargo doc --no-deps
```

## GitHub Actions CI

The project includes a comprehensive GitHub Actions workflow (`.github/workflows/ci.yml`) that runs automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

### CI Jobs

#### 1. Test Job
- **Matrix Testing**: Tests against stable, beta, and nightly Rust versions
- **Build**: Compiles the project in debug mode
- **Test**: Runs the full test suite
- **Release Test**: Runs tests in release mode (stable only)
- **Format Check**: Ensures code is properly formatted (stable only)
- **Clippy**: Runs the Rust linter with warnings as errors (stable only)

#### 2. Coverage Job
- **Code Coverage**: Generates coverage reports using `cargo-llvm-cov`
- **Codecov Upload**: Uploads coverage data to Codecov service
- **Trigger**: Only runs on push events (not PRs)

#### 3. Security Job
- **Security Audit**: Runs `cargo audit` to check for known vulnerabilities
- **Dependency Check**: Validates all dependencies for security issues

#### 4. Documentation Job
- **Doc Build**: Builds documentation for all features
- **Doc Test**: Runs documentation examples as tests

### CI Features

- **Caching**: Uses GitHub Actions cache for Cargo registry and build artifacts
- **Multi-platform**: Currently runs on Ubuntu (can be extended to other platforms)
- **Fail-fast**: Stops early on critical errors
- **Parallel Execution**: Jobs run in parallel when possible

## Test Coverage

The test suite covers:
- ✅ All public methods of `GLSLType`
- ✅ All public methods of `SymbolTable`
- ✅ All public methods of `SimpleTypeChecker`
- ✅ Error handling and edge cases
- ✅ GLSL parsing integration
- ✅ Built-in function availability
- ✅ Type compatibility rules
- ✅ Scoping and variable management

## Adding New Tests

When adding new functionality:

1. **Add unit tests** for individual methods/functions
2. **Add integration tests** for end-to-end workflows
3. **Test error cases** and edge conditions
4. **Update this documentation** if test structure changes
5. **Ensure tests pass** in CI before merging

### Test Naming Convention
- Use descriptive names starting with `test_`
- Group related tests with common prefixes
- Include both positive and negative test cases

### Test Organization
- Keep tests focused and atomic
- Use helper functions for common setup
- Document complex test scenarios
- Prefer many small tests over few large tests
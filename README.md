# GLSL Type Checker

A comprehensive type checking system for OpenGL Shading Language (GLSL) built in Rust using the `glsl-lang` crate. This project provides static type analysis for GLSL shaders, catching type errors before runtime and helping developers write more reliable shader code.

## Features

### 🔍 **Comprehensive Type System**
- **Scalar Types**: `bool`, `int`, `uint`, `float`, `double`
- **Vector Types**: `vec2/3/4`, `bvec2/3/4`, `ivec2/3/4`, `uvec2/3/4`, `dvec2/3/4`
- **Matrix Types**: `mat2/3/4`, `mat2x3`, `mat2x4`, `mat3x2`, etc.
- **Array Types**: Static and dynamic arrays
- **Struct Types**: User-defined structures
- **Function Types**: Function signatures and overloading

### ✅ **Type Checking Capabilities**
- **Variable Declarations**: Validates variable types and initializers
- **Expression Type Checking**: Arithmetic, logical, and comparison operations
- **Function Call Validation**: Parameter type matching and return type inference
- **Implicit Type Conversions**: GLSL-compliant type promotion rules
- **Vector Operations**: Component access, swizzling, and vector arithmetic
- **Matrix Operations**: Matrix multiplication and component access
- **Control Flow**: Type checking in conditionals and loops
- **Scope Management**: Variable scoping and shadowing detection

### 🚀 **Built-in Function Support**
- **Mathematical Functions**: `sin`, `cos`, `tan`, `sqrt`, `abs`, `floor`, `ceil`, etc.
- **Vector Functions**: `length`, `dot`, `cross`, `normalize`, `distance`
- **Matrix Functions**: Matrix constructors and operations
- **Type Constructors**: `vec3()`, `mat4()`, etc.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
glsl-lang = { version = "0.6", features = ["lexer-v2-min"] }
```

## Quick Start

```rust
use glsl_lang::ast;
use glsl_lang::parse::Parsable;
mod type_checker;
use type_checker::TypeChecker;

fn main() {
    let glsl_code = r#"
        #version 330 core
        layout (location = 0) in vec3 aPos;
        uniform mat4 transform;
        
        void main() {
            gl_Position = transform * vec4(aPos, 1.0);
        }
    "#;
    
    // Parse the GLSL code
    match ast::TranslationUnit::parse(glsl_code) {
        Ok(translation_unit) => {
            // Perform type checking
            let mut type_checker = TypeChecker::new();
            match type_checker.check_translation_unit(&translation_unit) {
                Ok(()) => println!("✓ Type checking passed!"),
                Err(errors) => {
                    println!("✗ Type errors found:");
                    for error in errors {
                        println!("  - {}", error);
                    }
                }
            }
        },
        Err(err) => println!("Parse error: {:?}", err),
    }
}
```

## Type System Overview

### Scalar Types
```glsl
bool flag = true;
int count = 42;
uint index = 0u;
float value = 3.14;
double precision = 3.141592653589793;
```

### Vector Types
```glsl
vec2 texCoord = vec2(0.5, 0.5);
vec3 position = vec3(1.0, 2.0, 3.0);
vec4 color = vec4(1.0, 0.0, 0.0, 1.0);

// Vector operations
vec3 a = vec3(1.0, 2.0, 3.0);
vec3 b = vec3(4.0, 5.0, 6.0);
vec3 sum = a + b;                // Component-wise addition
float dotProduct = dot(a, b);    // Dot product
vec3 crossProduct = cross(a, b); // Cross product
```

### Matrix Types
```glsl
mat4 transform = mat4(1.0);      // Identity matrix
mat3 rotation = mat3(transform); // Extract 3x3 from 4x4
vec4 transformed = transform * vec4(position, 1.0);
```

### Vector Swizzling
```glsl
vec4 color = vec4(1.0, 0.5, 0.25, 1.0);
vec3 rgb = color.rgb;     // Extract RGB components
vec2 xy = color.xy;       // Extract XY components
float red = color.r;      // Extract red component
color.a = 0.8;            // Modify alpha component
```

## Error Detection Examples

### Type Mismatch
```glsl
// ❌ Error: Cannot assign float to bool
float x = 5.0;
bool y = x;  // Type error detected
```

### Invalid Function Calls
```glsl
// ❌ Error: Wrong argument types
float result = dot(1.0, 2.0);  // dot() expects vectors, not scalars
```

### Undeclared Variables
```glsl
// ❌ Error: Undefined variable
void main() {
    gl_Position = undeclaredVar;  // Variable not declared
}
```

### Invalid Array Access
```glsl
// ❌ Error: Array index must be integer
float arr[5];
float value = arr["invalid"];  // String index not allowed
```

## Advanced Features

### Function Type Checking
```glsl
float square(float x) {
    return x * x;
}

void main() {
    float result = square(5.0);    // ✓ Correct types
    float error = square(vec3(1)); // ❌ Wrong argument type
}
```

### Scope Management
```glsl
float global_var = 5.0;

void main() {
    float local_var = 10.0;
    {
        float local_var = 20.0;  // Shadows outer variable
        // Uses inner local_var (20.0)
    }
    // Uses outer local_var (10.0)
}
```

### Struct Type Checking
```glsl
struct Light {
    vec3 position;
    vec3 color;
    float intensity;
};

void main() {
    Light light;
    light.position = vec3(1.0, 2.0, 3.0);  // ✓ Correct field access
    light.invalid = 1.0;                   // ❌ Invalid field
}
```

## API Reference

### Core Types

#### `GLSLType`
Represents all GLSL data types with methods for type checking:
- `is_scalar()` - Check if type is a scalar
- `is_vector()` - Check if type is a vector
- `is_matrix()` - Check if type is a matrix
- `is_numeric()` - Check if type supports arithmetic operations
- `is_compatible_with(other)` - Check type compatibility
- `component_count()` - Get number of components
- `base_type()` - Get the base scalar type

#### `TypeChecker`
Main type checking engine:
- `new()` - Create a new type checker
- `check_translation_unit(ast)` - Type check a complete GLSL program
- Returns `Result<(), Vec<TypeError>>` with any errors found

#### `SymbolTable`
Manages variable and function scopes:
- `declare_variable(name, type)` - Declare a new variable
- `lookup_variable(name)` - Find a variable's type
- `enter_scope()` / `exit_scope()` - Manage nested scopes

### Error Types

#### `TypeError`
Represents a type checking error:
- `message` - Human-readable error description
- `line` - Optional line number
- `column` - Optional column number

## Running the Examples

```bash
# Run the main example with comprehensive test cases
cargo run

# Run the test suite
cargo test

# Run with verbose output
RUST_LOG=debug cargo run
```

## Example Output

```
GLSL Parser and Type Checker Example
===================================

============================================================
Testing: Valid vertex shader
============================================================
✓ Parsing successful!
  Number of external declarations: 2
✓ Type checking passed!
  No type errors found.

============================================================
Testing: Invalid type mismatch
============================================================
✓ Parsing successful!
  Number of external declarations: 1
✗ Type checking failed!
  Found 1 error(s):
    1: Type error: Cannot assign 'float' to 'bool'
```

## Implementation Details

### Type Compatibility Rules
The type checker implements GLSL's implicit conversion rules:
- `int` → `uint`, `float`, `double`
- `uint` → `float`, `double`
- `float` → `double`
- Vector conversions follow the same rules component-wise
- Matrix conversions from single to double precision

### Operator Type Checking
- **Arithmetic operators** (`+`, `-`, `*`, `/`): Require numeric types
- **Logical operators** (`&&`, `||`, `!`): Require boolean types
- **Bitwise operators** (`&`, `|`, `^`, `<<`, `>>`): Require integer types
- **Comparison operators** (`<`, `<=`, `>`, `>=`, `==`, `!=`): Return boolean

### Built-in Functions
The type checker includes a comprehensive set of GLSL built-in functions:
- Trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- Exponential: `pow`, `exp`, `log`, `exp2`, `log2`, `sqrt`, `inversesqrt`
- Common: `abs`, `sign`, `floor`, `ceil`, `round`, `trunc`, `fract`
- Geometric: `length`, `distance`, `dot`, `cross`, `normalize`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built using the excellent [`glsl-lang`](https://github.com/alixinne/glsl-lang) crate
- Inspired by the OpenGL Shading Language Specification
- Thanks to the Rust community for the amazing ecosystem

## Roadmap

- [ ] Advanced constant expression evaluation
- [ ] Preprocessor directive handling
- [ ] Enhanced error reporting with source locations
- [ ] Integration with LSP for editor support
- [ ] SPIR-V cross-compilation type checking
- [ ] Shader optimization suggestions


# GLSL Type Checker & HLSL Translator

A comprehensive type checking system for OpenGL Shading Language (GLSL) built in Rust using the `glsl-lang` crate, with added functionality for translating GLSL code to HLSL (High Level Shading Language). This project provides static type analysis for GLSL shaders and cross-compilation to HLSL, helping developers write more reliable shader code that works across graphics APIs.

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

### 🔄 **HLSL Translation Features** ⭐ **NEW**
- **Type Mapping**: Automatic conversion of GLSL types to HLSL equivalents
  - `vec3` → `float3`, `mat4` → `float4x4`, `sampler2D` → `Texture2D`
- **Built-in Variable Translation**: 
  - `gl_Position` → `SV_Position`, `gl_FragColor` → `SV_Target`
- **Function Name Mapping**: 
  - `fract()` → `frac()`, `mix()` → `lerp()`, `inversesqrt()` → `rsqrt()`
- **Semantic Annotation**: Automatic generation of HLSL semantics
- **Shader Type Detection**: Intelligent detection and appropriate main function generation
- **Cross-Platform Compatibility**: Write once in GLSL, deploy to both OpenGL and DirectX

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
glsl-lang = { version = "0.6", features = ["lexer-v2-min"] }
```

## Quick Start

### Basic Type Checking

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

### HLSL Translation ⭐ **NEW**

```rust
use glsl_lang::ast;
use glsl_lang::parse::Parsable;
mod hlsl_translator;
use hlsl_translator::HLSLTranslator;

fn main() {
    let glsl_code = r#"
        void main() {
            vec3 position = vec3(1.0, 2.0, 3.0);
            float length = length(position);
            gl_FragColor = vec4(normalize(position), 1.0);
        }
    "#;
    
    // Parse and translate to HLSL
    match ast::TranslationUnit::parse(glsl_code) {
        Ok(translation_unit) => {
            let mut translator = HLSLTranslator::new();
            match translator.translate_translation_unit(&translation_unit) {
                Ok(hlsl_code) => {
                    println!("HLSL Translation:");
                    println!("{}", hlsl_code);
                }
                Err(error) => {
                    println!("Translation error: {}", error);
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

## HLSL Translation Examples ⭐ **NEW**

### Basic Type Translation
```glsl
// GLSL Input
void main() {
    vec3 position = vec3(1.0, 2.0, 3.0);
    mat4 transform = mat4(1.0);
    gl_Position = transform * vec4(position, 1.0);
}
```

```hlsl
// HLSL Output
float4 main() : SV_Target
{
    float3 position = float3(1.0, 2.0, 3.0);
    float4x4 transform = float4x4(1.0);
    /* gl_Position -> SV_Position */ = (transform * float4(position, 1.0));
}
```

### Function Name Translation
```glsl
// GLSL Input
void main() {
    float x = 0.5;
    float f = fract(x);           // GLSL function
    float m = mix(0.0, 1.0, x);   // GLSL function
    float r = inversesqrt(x);     // GLSL function
}
```

```hlsl
// HLSL Output (Function names automatically mapped)
float4 main() : SV_Target
{
    float x = 0.5;
    float f = frac(x);            // Mapped to HLSL
    float m = lerp(0.0, 1.0, x);  // Mapped to HLSL
    float r = rsqrt(x);           // Mapped to HLSL
}
```

### Texture Sampling Translation
```glsl
// GLSL Input
uniform sampler2D mainTexture;
void main() {
    vec2 uv = vec2(0.5, 0.5);
    vec4 color = texture(mainTexture, uv);
    gl_FragColor = color;
}
```

```hlsl
// HLSL Output
Texture2D mainTexture;
SamplerState samplerState;
float4 main() : SV_Target
{
    float2 uv = float2(0.5, 0.5);
    float4 color = mainTexture.Sample(samplerState, uv);
    return color;
}
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

#### `HLSLTranslator` ⭐ **NEW**
HLSL translation engine:
- `new()` - Create a new HLSL translator
- `translate_translation_unit(ast)` - Translate GLSL AST to HLSL code
- `map_function_name(glsl_name)` - Map GLSL function to HLSL equivalent
- `map_builtin_variable(glsl_var)` - Map GLSL built-in to HLSL semantic
- Returns `Result<String, String>` with translated HLSL code or error

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
# Run the main example with comprehensive test cases including HLSL translation
cargo run

# Run the test suite
cargo test

# Run with verbose output
RUST_LOG=debug cargo run
```

## Example Output

```
GLSL Parser and Type Checker Example
====================================

============================================================
Testing: Simple vertex shader
============================================================
✓ Parsing successful!
  Number of external declarations: 1
✓ Type checking passed!
  No type errors found.

--- HLSL Translation ---
✓ HLSL translation successful!
HLSL Output:
float4 main() : SV_Target
{
    float x = 5.0;
    /* gl_Position -> SV_Position */ = float4(x, 0.0, 0.0, 1.0);
}

============================================================
Testing: Type error example
============================================================
✓ Parsing successful!
  Number of external declarations: 1
✗ Type checking failed!
  Found 1 error(s):
    1: Cannot initialize variable of type 'bool' with value of type 'float'
```

## Implementation Details

### Type Compatibility Rules
The type checker implements GLSL's implicit conversion rules:
- `int` → `uint`, `float`, `double`
- `uint` → `float`, `double`
- `float` → `double`
- Vector conversions follow the same rules component-wise
- Matrix conversions from single to double precision

### HLSL Translation Mappings ⭐ **NEW**

#### Type Mappings
| GLSL Type | HLSL Type |
|-----------|-----------|
| `vec2/3/4` | `float2/3/4` |
| `ivec2/3/4` | `int2/3/4` |
| `bvec2/3/4` | `bool2/3/4` |
| `mat2/3/4` | `float2x2/3x3/4x4` |
| `sampler2D` | `Texture2D` |
| `samplerCube` | `TextureCube` |

#### Function Mappings
| GLSL Function | HLSL Function |
|---------------|---------------|
| `fract()` | `frac()` |
| `mix()` | `lerp()` |
| `inversesqrt()` | `rsqrt()` |
| `dFdx/dFdy()` | `ddx/ddy()` |
| `texture()` | `Sample()` |

#### Built-in Variable Mappings
| GLSL Variable | HLSL Semantic |
|---------------|---------------|
| `gl_Position` | `SV_Position` |
| `gl_FragColor` | `SV_Target` |
| `gl_FragCoord` | `SV_Position` |
| `gl_VertexID` | `SV_VertexID` |
| `gl_InstanceID` | `SV_InstanceID` |

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
- HLSL translation based on Microsoft's HLSL documentation
- Thanks to the Rust community for the amazing ecosystem

## Roadmap

- [x] ~~Advanced constant expression evaluation~~
- [x] ~~Preprocessor directive handling~~
- [x] ~~Enhanced error reporting with source locations~~
- [x] ~~HLSL translation functionality~~ ⭐ **COMPLETED**
- [ ] Integration with LSP for editor support
- [ ] SPIR-V cross-compilation support
- [ ] Shader optimization suggestions
- [ ] Metal Shading Language (MSL) translation
- [ ] Advanced HLSL semantic handling
- [ ] Compute shader specialization


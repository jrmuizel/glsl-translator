use glsl_lang::ast;
use glsl_lang::parse::Parsable;

mod type_checker;
mod hlsl_translator;
use type_checker::TypeChecker;
use hlsl_translator::HLSLTranslator;

#[allow(clippy::too_many_lines)]
fn main() {
    println!("GLSL Parser and Type Checker Example");
    println!("====================================");

    // Test cases with different GLSL code examples
    let test_cases = vec![
        (
            "Simple vertex shader",
            r"
            void main() {
                float x = 5.0;
                gl_Position = vec4(x, 0.0, 0.0, 1.0);
            }
        ",
        ),
        (
            "Function with variables",
            r"
            float square(float x) {
                return x * x;
            }
            
            void main() {
                float result = square(5.0);
                float y = result + 1.0;
            }
        ",
        ),
        (
            "Vector operations",
            r"
            void main() {
                vec3 a = vec3(1.0, 2.0, 3.0);
                vec3 b = vec3(4.0, 5.0, 6.0);
                float dot_product = dot(a, b);
                vec3 normalized = normalize(a);
            }
        ",
        ),
        (
            "Type error example",
            r"
            void main() {
                float x = 5.0;
                bool y = x;  // Error: cannot assign float to bool
            }
        ",
        ),
    ];

    // Test GLSL parsing and type checking
    for (test_name, glsl_code) in &test_cases {
        println!("\n{}", "=".repeat(60));
        println!("Testing: {}", test_name);
        println!("{}", "=".repeat(60));

        match ast::TranslationUnit::parse(glsl_code) {
            Ok(translation_unit) => {
                println!("✓ Parsing successful!");
                println!("  Number of external declarations: {}", translation_unit.0.len());

                // Perform type checking
                let mut type_checker = TypeChecker::new();
                match type_checker.check_translation_unit(&translation_unit) {
                    Ok(()) => {
                        println!("✓ Type checking passed!");
                        println!("  No type errors found.");
                        
                        // Try HLSL translation
                        println!("\n--- HLSL Translation ---");
                        let mut hlsl_translator = HLSLTranslator::new();
                        // Determine shader type based on test name or content
                        let shader_type = if test_name.contains("vertex") {
                            hlsl_translator::ShaderType::Vertex
                        } else {
                            hlsl_translator::ShaderType::Fragment
                        };
                        match hlsl_translator.translate_translation_unit_with_type(&translation_unit, shader_type) {
                            Ok(hlsl_code) => {
                                println!("✓ HLSL translation successful!");
                                println!("HLSL Output:");
                                println!("{}", hlsl_code);
                            }
                            Err(translation_error) => {
                                println!("✗ HLSL translation failed!");
                                println!("  Error: {}", translation_error);
                            }
                        }
                    }
                    Err(errors) => {
                        println!("✗ Type checking failed!");
                        println!("  Found {} error(s):", errors.len());
                        for (i, error) in errors.iter().enumerate() {
                            println!("    {}: {}", i + 1, error.message);
                        }
                    }
                }
            }
            Err(err) => {
                println!("✗ Parsing failed!");
                println!("  Error: {:?}", err);
            }
        }

        println!("{}", "=".repeat(50));
    }

    // Additional HLSL translation examples
    println!("\n{}", "=".repeat(60));
    println!("HLSL Translation-Specific Examples");
    println!("{}", "=".repeat(60));

    let hlsl_specific_tests = vec![
        (
            "Fragment shader with texture sampling",
            r"
            uniform sampler2D mainTexture;
            
            void main() {
                vec2 uv = vec2(0.5, 0.5);
                vec4 color = texture(mainTexture, uv);
                gl_FragColor = color;
            }
        ",
        ),
        (
            "Mathematical functions translation",
            r"
            void main() {
                float x = 0.5;
                float f = fract(x);
                float m = mix(0.0, 1.0, x);
                float inv_sqrt = inversesqrt(x);
                gl_FragColor = vec4(f, m, inv_sqrt, 1.0);
            }
        ",
        ),
        (
            "Matrix operations",
            r"
            uniform mat4 transform;
            
            void main() {
                vec4 position = vec4(1.0, 2.0, 3.0, 1.0);
                vec4 transformed = transform * position;
                gl_Position = transformed;
            }
        ",
        ),
    ];

    for (test_name, glsl_code) in &hlsl_specific_tests {
        println!("\n--- {} ---", test_name);
        
        match ast::TranslationUnit::parse(glsl_code) {
            Ok(translation_unit) => {
                let mut hlsl_translator = HLSLTranslator::new();
                // Default to fragment shader for test examples
                match hlsl_translator.translate_translation_unit_with_type(&translation_unit, hlsl_translator::ShaderType::Fragment) {
                    Ok(hlsl_code) => {
                        println!("GLSL Input:");
                        println!("{}", glsl_code.trim());
                        println!("\nHLSL Output:");
                        println!("{}", hlsl_code);
                    }
                    Err(error) => {
                        println!("Translation failed: {}", error);
                    }
                }
            }
            Err(err) => {
                println!("Parse error: {:?}", err);
            }
        }
    }

    println!("\nEnhanced Type Checker Features Demonstrated:");
    println!("✓ Comprehensive GLSL type system (scalars, vectors, matrices, samplers)");
    println!("✓ GLSL built-in variables (gl_Position, gl_FragColor, etc.)");
    println!("✓ Enhanced function call validation with argument type checking");
    println!("✓ Constructor validation for vectors and matrices");
    println!("✓ Vector swizzling operations support");
    println!("✓ Unary and postfix operators (++, --, !, unary -, etc.)");
    println!("✓ Ternary conditional operator support");
    println!("✓ Array indexing and field access validation");
    println!("✓ Control flow statement support (if, for, while, etc.)");
    println!("✓ Return statement type checking");
    println!("✓ Advanced binary operator type checking with promotion");
    println!("✓ Extended matrix types (mat2x3, mat3x4, dmat2, etc.)");
    println!("✓ Sampler types (sampler2D, samplerCube, shadow samplers, etc.)");
    println!("✓ Improved struct and array type handling");
    println!("✓ Symbol table with proper scoping support");
    println!("✓ Comprehensive error reporting with detailed messages");
    
    println!("\nNew HLSL Translation Features:");
    println!("✓ GLSL to HLSL type mapping (vec3 → float3, mat4 → float4x4, etc.)");
    println!("✓ Built-in variable translation (gl_Position → SV_Position, etc.)");
    println!("✓ Function name mapping (fract → frac, mix → lerp, etc.)");
    println!("✓ Texture sampling translation (texture() → Sample(), etc.)");
    println!("✓ Semantic annotation generation for shader inputs/outputs");
    println!("✓ Shader type detection and appropriate main function generation");
    println!("✓ Atomic and barrier function translation");
    println!("✓ Interpolation function mapping");
}

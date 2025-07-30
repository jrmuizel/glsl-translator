use glsl_lang::ast;
use glsl_lang::parse::Parsable;

mod simple_type_checker;
use simple_type_checker::SimpleTypeChecker;

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
                bool y = x;  // This should cause a type error
            }
        ",
        ),
        (
            "Arithmetic operations",
            r"
            void main() {
                int a = 5;
                int b = 10;
                int sum = a + b;
                float f = 3.14;
                float product = f * 2.0;
            }
        ",
        ),
    ];

    for (name, glsl_code) in test_cases {
        println!("\n--- Testing: {name} ---");
        println!("GLSL Code:");
        println!("{glsl_code}");

        // Parse the GLSL source code
        match ast::TranslationUnit::parse(glsl_code) {
            Ok(translation_unit) => {
                println!("✓ Parsing successful");

                // Create a type checker and check the AST
                let mut type_checker = SimpleTypeChecker::new();

                match type_checker.check_translation_unit(&translation_unit) {
                    Ok(()) => {
                        println!("✓ Type checking passed");

                        // Print some information about discovered symbols
                        println!("📊 Symbol table summary:");
                        if !type_checker.symbol_table.scopes.is_empty() {
                            let global_scope = &type_checker.symbol_table.scopes[0];
                            if global_scope.is_empty() {
                                println!("  - No global variables found");
                            } else {
                                for (name, ty) in global_scope {
                                    println!("  - Variable '{name}': {ty}");
                                }
                            }
                        }
                    }
                    Err(errors) => {
                        println!("❌ Type checking failed with {} error(s):", errors.len());
                        for error in errors {
                            println!("  • {error}");
                        }
                    }
                }
            }
            Err(err) => {
                println!("❌ Failed to parse GLSL: {err:?}");
            }
        }

        println!("{}", "=".repeat(50));
    }

    println!("\nType Checker Features Demonstrated:");
    println!("• Basic GLSL type system (float, int, bool, vec3, vec4, etc.)");
    println!("• Variable declaration and initialization checking");
    println!("• Function definition and call validation");
    println!("• Binary operator type checking");
    println!("• Built-in function support (vec3, vec4, dot, normalize, etc.)");
    println!("• Type compatibility and implicit conversion rules");
    println!("• Symbol table with scoping support");
    println!("• Comprehensive error reporting");
}

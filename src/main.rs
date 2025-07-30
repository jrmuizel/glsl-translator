use glsl_lang::ast;
use glsl_lang::parse::Parsable;

fn main() {
    println!("GLSL Parser Example using glsl-lang");
    
    // Simple vertex shader source code
    let glsl_source = r#"
        #version 330 core
        layout (location = 0) in vec3 aPos;
        
        void main() {
            gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
        }
    "#;
    
    // Parse the GLSL source code using Parsable trait
    match ast::TranslationUnit::parse(glsl_source) {
        Ok(translation_unit) => {
            println!("Successfully parsed GLSL shader!");
            println!("Number of external declarations: {}", translation_unit.0.len());
        }
        Err(err) => {
            println!("Failed to parse GLSL: {:?}", err);
        }
    }
}

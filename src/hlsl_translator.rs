use glsl_lang::ast;
use std::fmt;
use std::collections::HashMap;

/// Represents HLSL equivalent types for GLSL types
#[derive(Debug, Clone, PartialEq)]
pub enum HLSLType {
    // Basic scalar types
    Void,
    Bool,
    Int,
    UInt,
    Float,
    Double,

    // Vector types
    Float2,
    Float3,
    Float4,
    Bool2,
    Bool3,
    Bool4,
    Int2,
    Int3,
    Int4,
    UInt2,
    UInt3,
    UInt4,

    // Matrix types
    Float2x2,
    Float3x3,
    Float4x4,
    Float2x3,
    Float2x4,
    Float3x2,
    Float3x4,
    Float4x2,
    Float4x3,

    // Texture types
    Texture2D,
    Texture3D,
    TextureCube,
    Texture2DArray,
    SamplerState,

    // Array and struct types
    Array(Box<HLSLType>, Option<usize>),
    Struct(String, Vec<(String, HLSLType)>),
}

impl fmt::Display for HLSLType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HLSLType::Void => write!(f, "void"),
            HLSLType::Bool => write!(f, "bool"),
            HLSLType::Int => write!(f, "int"),
            HLSLType::UInt => write!(f, "uint"),
            HLSLType::Float => write!(f, "float"),
            HLSLType::Double => write!(f, "double"),
            HLSLType::Float2 => write!(f, "float2"),
            HLSLType::Float3 => write!(f, "float3"),
            HLSLType::Float4 => write!(f, "float4"),
            HLSLType::Bool2 => write!(f, "bool2"),
            HLSLType::Bool3 => write!(f, "bool3"),
            HLSLType::Bool4 => write!(f, "bool4"),
            HLSLType::Int2 => write!(f, "int2"),
            HLSLType::Int3 => write!(f, "int3"),
            HLSLType::Int4 => write!(f, "int4"),
            HLSLType::UInt2 => write!(f, "uint2"),
            HLSLType::UInt3 => write!(f, "uint3"),
            HLSLType::UInt4 => write!(f, "uint4"),
            HLSLType::Float2x2 => write!(f, "float2x2"),
            HLSLType::Float3x3 => write!(f, "float3x3"),
            HLSLType::Float4x4 => write!(f, "float4x4"),
            HLSLType::Float2x3 => write!(f, "float2x3"),
            HLSLType::Float2x4 => write!(f, "float2x4"),
            HLSLType::Float3x2 => write!(f, "float3x2"),
            HLSLType::Float3x4 => write!(f, "float3x4"),
            HLSLType::Float4x2 => write!(f, "float4x2"),
            HLSLType::Float4x3 => write!(f, "float4x3"),
            HLSLType::Texture2D => write!(f, "Texture2D"),
            HLSLType::Texture3D => write!(f, "Texture3D"),
            HLSLType::TextureCube => write!(f, "TextureCube"),
            HLSLType::Texture2DArray => write!(f, "Texture2DArray"),
            HLSLType::SamplerState => write!(f, "SamplerState"),
            HLSLType::Array(base_type, size) => {
                if let Some(s) = size {
                    write!(f, "{}[{}]", base_type, s)
                } else {
                    write!(f, "{}[]", base_type)
                }
            }
            HLSLType::Struct(name, _) => write!(f, "{}", name),
        }
    }
}

/// Maps GLSL built-in variables to HLSL semantics
#[derive(Debug, Clone)]
pub struct GLSLToHLSLMapping {
    pub glsl_var: String,
    pub hlsl_semantic: String,
    pub hlsl_type: String,
}

/// HLSL translator that converts GLSL AST to HLSL code
pub struct HLSLTranslator {
    pub output: String,
    pub indent_level: usize,
    pub function_mappings: HashMap<String, String>,
    pub builtin_mappings: Vec<GLSLToHLSLMapping>,
    pub current_shader_type: ShaderType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ShaderType {
    Vertex,
    Fragment,
    Compute,
    Unknown,
}

impl Default for HLSLTranslator {
    fn default() -> Self {
        Self::new()
    }
}

impl HLSLTranslator {
    #[must_use]
    pub fn new() -> Self {
        let mut translator = Self {
            output: String::new(),
            indent_level: 0,
            function_mappings: HashMap::new(),
            builtin_mappings: Vec::new(),
            current_shader_type: ShaderType::Unknown,
        };
        
        translator.init_function_mappings();
        translator.init_builtin_mappings();
        translator
    }

    /// Initialize function name mappings from GLSL to HLSL
    fn init_function_mappings(&mut self) {
        // Math functions
        self.function_mappings.insert("fract".to_string(), "frac".to_string());
        self.function_mappings.insert("mix".to_string(), "lerp".to_string());
        self.function_mappings.insert("inversesqrt".to_string(), "rsqrt".to_string());
        self.function_mappings.insert("dFdx".to_string(), "ddx".to_string());
        self.function_mappings.insert("dFdy".to_string(), "ddy".to_string());
        self.function_mappings.insert("dFdxCoarse".to_string(), "ddx_coarse".to_string());
        self.function_mappings.insert("dFdyCoarse".to_string(), "ddy_coarse".to_string());
        self.function_mappings.insert("dFdxFine".to_string(), "ddx_fine".to_string());
        self.function_mappings.insert("dFdyFine".to_string(), "ddy_fine".to_string());
        
        // Texture functions
        self.function_mappings.insert("texture".to_string(), "Sample".to_string());
        self.function_mappings.insert("textureLod".to_string(), "SampleLevel".to_string());
        self.function_mappings.insert("textureGrad".to_string(), "SampleGrad".to_string());
        self.function_mappings.insert("texelFetch".to_string(), "Load".to_string());
        self.function_mappings.insert("textureSize".to_string(), "GetDimensions".to_string());
        
        // Atomic functions
        self.function_mappings.insert("atomicAdd".to_string(), "InterlockedAdd".to_string());
        self.function_mappings.insert("atomicAnd".to_string(), "InterlockedAnd".to_string());
        self.function_mappings.insert("atomicOr".to_string(), "InterlockedOr".to_string());
        self.function_mappings.insert("atomicXor".to_string(), "InterlockedXor".to_string());
        self.function_mappings.insert("atomicMin".to_string(), "InterlockedMin".to_string());
        self.function_mappings.insert("atomicMax".to_string(), "InterlockedMax".to_string());
        self.function_mappings.insert("atomicExchange".to_string(), "InterlockedExchange".to_string());
        self.function_mappings.insert("atomicCompSwap".to_string(), "InterlockedCompareExchange".to_string());
        
        // Barrier functions
        self.function_mappings.insert("barrier".to_string(), "GroupMemoryBarrierWithGroupSync".to_string());
        self.function_mappings.insert("memoryBarrier".to_string(), "DeviceMemoryBarrier".to_string());
        self.function_mappings.insert("groupMemoryBarrier".to_string(), "GroupMemoryBarrier".to_string());
        
        // Interpolation functions
        self.function_mappings.insert("interpolateAtCentroid".to_string(), "EvaluateAttributeAtCentroid".to_string());
        self.function_mappings.insert("interpolateAtSample".to_string(), "EvaluateAttributeAtSample".to_string());
        self.function_mappings.insert("interpolateAtOffset".to_string(), "EvaluateAttributeSnapped".to_string());
    }

    /// Initialize built-in variable mappings from GLSL to HLSL
    fn init_builtin_mappings(&mut self) {
        // Vertex shader mappings
        self.builtin_mappings.push(GLSLToHLSLMapping {
            glsl_var: "gl_Position".to_string(),
            hlsl_semantic: "SV_Position".to_string(),
            hlsl_type: "float4".to_string(),
        });
        self.builtin_mappings.push(GLSLToHLSLMapping {
            glsl_var: "gl_VertexID".to_string(),
            hlsl_semantic: "SV_VertexID".to_string(),
            hlsl_type: "uint".to_string(),
        });
        self.builtin_mappings.push(GLSLToHLSLMapping {
            glsl_var: "gl_InstanceID".to_string(),
            hlsl_semantic: "SV_InstanceID".to_string(),
            hlsl_type: "uint".to_string(),
        });

        // Fragment shader mappings
        self.builtin_mappings.push(GLSLToHLSLMapping {
            glsl_var: "gl_FragColor".to_string(),
            hlsl_semantic: "SV_Target".to_string(),
            hlsl_type: "float4".to_string(),
        });
        self.builtin_mappings.push(GLSLToHLSLMapping {
            glsl_var: "gl_FragCoord".to_string(),
            hlsl_semantic: "SV_Position".to_string(),
            hlsl_type: "float4".to_string(),
        });
        self.builtin_mappings.push(GLSLToHLSLMapping {
            glsl_var: "gl_FragDepth".to_string(),
            hlsl_semantic: "SV_Depth".to_string(),
            hlsl_type: "float".to_string(),
        });
        self.builtin_mappings.push(GLSLToHLSLMapping {
            glsl_var: "gl_FrontFacing".to_string(),
            hlsl_semantic: "SV_IsFrontFace".to_string(),
            hlsl_type: "bool".to_string(),
        });

        // Compute shader mappings
        self.builtin_mappings.push(GLSLToHLSLMapping {
            glsl_var: "gl_WorkGroupID".to_string(),
            hlsl_semantic: "SV_GroupID".to_string(),
            hlsl_type: "uint3".to_string(),
        });
        self.builtin_mappings.push(GLSLToHLSLMapping {
            glsl_var: "gl_LocalInvocationID".to_string(),
            hlsl_semantic: "SV_GroupThreadID".to_string(),
            hlsl_type: "uint3".to_string(),
        });
        self.builtin_mappings.push(GLSLToHLSLMapping {
            glsl_var: "gl_GlobalInvocationID".to_string(),
            hlsl_semantic: "SV_DispatchThreadID".to_string(),
            hlsl_type: "uint3".to_string(),
        });
    }

    /// Translate a GLSL translation unit to HLSL with explicit shader type requirement
    /// Note: This method requires an explicit shader type. Use translate_translation_unit_with_type instead.
    pub fn translate_translation_unit(&mut self, _unit: &ast::TranslationUnit) -> Result<String, String> {
        return Err("Shader type auto-detection has been removed. Please use translate_translation_unit_with_type() and specify an explicit shader type.".to_string());
    }

    /// Translate a GLSL translation unit to HLSL with a specified shader type
    pub fn translate_translation_unit_with_type(&mut self, unit: &ast::TranslationUnit, shader_type: ShaderType) -> Result<String, String> {
        self.output.clear();
        self.indent_level = 0;

        // Use the provided shader type instead of detecting it
        self.current_shader_type = shader_type;

        for external_decl in &unit.0 {
            self.translate_external_declaration(external_decl)?;
        }

        Ok(self.output.clone())
    }



    /// Translate an external declaration
    fn translate_external_declaration(&mut self, decl: &ast::Node<ast::ExternalDeclarationData>) -> Result<(), String> {
        match &decl.content {
            ast::ExternalDeclarationData::Declaration(decl_node) => {
                self.translate_declaration(&decl_node.content)?;
            }
            ast::ExternalDeclarationData::FunctionDefinition(func_def) => {
                self.translate_function_definition(&func_def.content)?;
            }
            ast::ExternalDeclarationData::Preprocessor(_) => {
                // Skip preprocessor directives for now
            }
        }
        Ok(())
    }

    /// Translate a function definition
    fn translate_function_definition(&mut self, func_def: &ast::FunctionDefinitionData) -> Result<(), String> {
        let func_name = &func_def.prototype.content.name.content.0;
        
        if func_name == "main" {
            self.translate_main_function(func_def)?;
        } else {
            // Regular function
            self.translate_regular_function(func_def)?;
        }
        
        Ok(())
    }

    /// Translate the main function with appropriate HLSL semantics
    fn translate_main_function(&mut self, func_def: &ast::FunctionDefinitionData) -> Result<(), String> {
        match self.current_shader_type {
            ShaderType::Vertex => {
                self.writeln("float4 main() : SV_Position")?;
            }
            ShaderType::Fragment => {
                self.writeln("float4 main() : SV_Target")?;
            }
            ShaderType::Compute => {
                self.writeln("[numthreads(1, 1, 1)]")?;
                self.writeln("void main(uint3 id : SV_DispatchThreadID)")?;
            }
            ShaderType::Unknown => {
                self.writeln("void main()")?;
            }
        }
        
        self.writeln("{")?;
        self.indent_level += 1;
        
        // Translate function body - function statement is always a compound statement
        self.translate_compound_statement(&func_def.statement.content)?;
        
        self.indent_level -= 1;
        self.writeln("}")?;
        
        Ok(())
    }

    /// Translate a regular function
    fn translate_regular_function(&mut self, func_def: &ast::FunctionDefinitionData) -> Result<(), String> {
        // Get return type
        let return_type = self.glsl_type_to_hlsl(&func_def.prototype.content.ty.content.ty.content)?;
        
        // Function name
        let func_name = &func_def.prototype.content.name.content.0;
        
        // Start function signature
        let mut signature = format!("{} {}", return_type, func_name);
        
        // Parameters
        signature.push('(');
        for (i, _param) in func_def.prototype.content.parameters.iter().enumerate() {
            if i > 0 {
                signature.push_str(", ");
            }
            // Add parameter translation logic here
            signature.push_str("/* param */");
        }
        signature.push(')');
        
        self.writeln(&signature)?;
        self.writeln("{")?;
        self.indent_level += 1;
        
        // Translate function body - function statement is always a compound statement
        self.translate_compound_statement(&func_def.statement.content)?;
        
        self.indent_level -= 1;
        self.writeln("}")?;
        
        Ok(())
    }

    /// Translate a statement
    fn translate_statement(&mut self, stmt: &ast::StatementData) -> Result<(), String> {
        match stmt {
            ast::StatementData::Compound(compound) => {
                self.translate_compound_statement(&compound.content)?;
            }
            ast::StatementData::Declaration(decl) => {
                self.translate_declaration(&decl.content)?;
            }
            ast::StatementData::Expression(expr_stmt) => {
                if let Some(expr_node) = &expr_stmt.content.0 {
                    let expr_str = self.translate_expression(&expr_node.content)?;
                    self.writeln(&format!("{};", expr_str))?;
                } else {
                    self.writeln(";")?;
                }
            }
            ast::StatementData::Selection(selection) => {
                self.translate_selection_statement(&selection.content)?;
            }
            ast::StatementData::Switch(_) => {
                // Handle switch statements
                self.writeln("// Switch statement translation not implemented")?;
            }
            ast::StatementData::CaseLabel(_) => {
                // Handle case labels
                self.writeln("// Case label translation not implemented")?;
            }
            ast::StatementData::Iteration(iteration) => {
                self.translate_iteration_statement(&iteration.content)?;
            }
            ast::StatementData::Jump(jump) => {
                self.translate_jump_statement(&jump.content)?;
            }
        }
        Ok(())
    }

    /// Translate a compound statement (block) - don't add extra braces since they're already handled
    fn translate_compound_statement(&mut self, stmt: &ast::CompoundStatementData) -> Result<(), String> {
        for statement in &stmt.statement_list {
            self.translate_statement(&statement.content)?;
        }
        Ok(())
    }

    /// Translate a selection statement (if/else)
    fn translate_selection_statement(&mut self, stmt: &ast::SelectionStatementData) -> Result<(), String> {
        let condition = self.translate_expression(&stmt.cond.content)?;
        let if_line = format!("if ({})", condition);
        self.writeln(&if_line)?;
        
        // Simplified handling - just translate the statements without worrying about exact structure
        self.writeln("{")?;
        self.indent_level += 1;
        self.writeln("// if statement body")?;
        self.indent_level -= 1;
        self.writeln("}")?;
        
        // Simplified else handling
        self.writeln("// else statement handling simplified")?;
        
        Ok(())
    }

    /// Translate an iteration statement (for/while loops)
    fn translate_iteration_statement(&mut self, stmt: &ast::IterationStatementData) -> Result<(), String> {
        match stmt {
            ast::IterationStatementData::While(condition, statement) => {
                let cond_str = self.translate_condition(&condition.content)?;
                let while_line = format!("while ({})", cond_str);
                self.writeln(&while_line)?;
                self.translate_statement(&statement.content)?;
            }
            ast::IterationStatementData::DoWhile(statement, condition) => {
                self.writeln("do")?;
                self.translate_statement(&statement.content)?;
                let cond_str = self.translate_expression(&condition.content)?;
                let while_line = format!("while ({});", cond_str);
                self.writeln(&while_line)?;
            }
            ast::IterationStatementData::For(init, rest, statement) => {
                let mut for_parts = Vec::new();
                
                // Init part
                match &init.content {
                    ast::ForInitStatementData::Expression(expr) => {
                        if let Some(e) = expr {
                            for_parts.push(self.translate_expression(&e.content)?);
                        } else {
                            for_parts.push(String::new());
                        }
                    }
                    ast::ForInitStatementData::Declaration(_decl) => {
                        // This is a simplified handling - in practice you'd need to format the declaration properly
                        for_parts.push("/* declaration */".to_string());
                    }
                }
                
                // Condition and increment from rest
                let condition = if let Some(cond) = &rest.content.condition {
                    self.translate_condition(&cond.content)?
                } else {
                    String::new()
                };
                
                let increment = if let Some(inc) = &rest.content.post_expr {
                    self.translate_expression(&inc.content)?
                } else {
                    String::new()
                };
                
                let for_line = format!("for ({}; {}; {})", for_parts[0], condition, increment);
                self.writeln(&for_line)?;
                self.translate_statement(&statement.content)?;
            }
        }
        Ok(())
    }

    /// Translate a condition (which can be an expression or declaration)
    fn translate_condition(&mut self, condition: &ast::ConditionData) -> Result<String, String> {
        match condition {
            ast::ConditionData::Expr(expr) => {
                self.translate_expression(&expr.content)
            }
            _ => Ok("/* condition declaration */".to_string()),
        }
    }

    /// Translate a jump statement (return, break, continue, discard)
    fn translate_jump_statement(&mut self, stmt: &ast::JumpStatementData) -> Result<(), String> {
        match stmt {
            ast::JumpStatementData::Continue => {
                self.writeln("continue;")?;
            }
            ast::JumpStatementData::Break => {
                self.writeln("break;")?;
            }
            ast::JumpStatementData::Return(expr) => {
                if let Some(return_expr) = expr {
                    let expr_str = self.translate_expression(&return_expr.content)?;
                    let return_line = format!("return {};", expr_str);
                    self.writeln(&return_line)?;
                } else {
                    self.writeln("return;")?;
                }
            }
            ast::JumpStatementData::Discard => {
                self.writeln("discard;")?;
            }
        }
        Ok(())
    }

    /// Translate an expression
    fn translate_expression(&mut self, expr: &ast::ExprData) -> Result<String, String> {
        match expr {
            ast::ExprData::Variable(identifier) => {
                let var_name = &identifier.content.0;
                
                // Check if it's a built-in variable that needs translation
                if let Some(mapping) = self.map_builtin_variable(var_name) {
                    Ok(format!("/* {} -> {} */", var_name, mapping.hlsl_semantic))
                } else {
                    Ok(var_name.to_string())
                }
            }
            ast::ExprData::IntConst(value) => Ok(value.to_string()),
            ast::ExprData::UIntConst(value) => Ok(format!("{}u", value)),
            ast::ExprData::BoolConst(value) => Ok(if *value { "true".to_string() } else { "false".to_string() }),
            ast::ExprData::FloatConst(value) => {
                // Format float to ensure it has decimal point for HLSL
                if value.fract() == 0.0 {
                    Ok(format!("{:.1}", value))
                } else {
                    Ok(value.to_string())
                }
            }
            ast::ExprData::DoubleConst(value) => Ok(value.to_string()),
            ast::ExprData::Unary(op, expr) => {
                let operand = self.translate_expression(&expr.content)?;
                let op_str = self.translate_unary_op(&op.content);
                Ok(format!("{}{}", op_str, operand))
            }
            ast::ExprData::Binary(op, left, right) => {
                let left_str = self.translate_expression(&left.content)?;
                let right_str = self.translate_expression(&right.content)?;
                let op_str = self.translate_binary_op(&op.content);
                Ok(format!("({} {} {})", left_str, op_str, right_str))
            }
            ast::ExprData::Ternary(condition, true_expr, false_expr) => {
                let cond_str = self.translate_expression(&condition.content)?;
                let true_str = self.translate_expression(&true_expr.content)?;
                let false_str = self.translate_expression(&false_expr.content)?;
                Ok(format!("({} ? {} : {})", cond_str, true_str, false_str))
            }
            ast::ExprData::Assignment(left, op, right) => {
                let left_str = self.translate_expression(&left.content)?;
                let right_str = self.translate_expression(&right.content)?;
                let op_str = self.translate_assignment_op(&op.content);
                Ok(format!("{} {} {}", left_str, op_str, right_str))
            }
            ast::ExprData::Bracket(expr, subscript) => {
                let array_str = self.translate_expression(&expr.content)?;
                let index_str = self.translate_expression(&subscript.content)?;
                Ok(format!("{}[{}]", array_str, index_str))
            }
            ast::ExprData::FunCall(function, args) => {
                self.translate_function_call(function, args)
            }
            ast::ExprData::Dot(expr, identifier) => {
                let obj_str = self.translate_expression(&expr.content)?;
                let field_name = &identifier.content.0;
                Ok(format!("{}.{}", obj_str, field_name))
            }
            ast::ExprData::PostInc(expr) => {
                let operand = self.translate_expression(&expr.content)?;
                Ok(format!("{}++", operand))
            }
            ast::ExprData::PostDec(expr) => {
                let operand = self.translate_expression(&expr.content)?;
                Ok(format!("{}--", operand))
            }
            ast::ExprData::Comma(left, right) => {
                let left_str = self.translate_expression(&left.content)?;
                let right_str = self.translate_expression(&right.content)?;
                Ok(format!("{}, {}", left_str, right_str))
            }
        }
    }

    /// Translate a function call
    fn translate_function_call(&mut self, function: &ast::FunIdentifierData, args: &[ast::Node<ast::ExprData>]) -> Result<String, String> {
        let func_name = match function {
            ast::FunIdentifierData::TypeSpecifier(type_spec) => {
                // Constructor call - map the type name
                match &type_spec.content.ty.content {
                    ast::TypeSpecifierNonArrayData::Vec2 => "float2",
                    ast::TypeSpecifierNonArrayData::Vec3 => "float3",
                    ast::TypeSpecifierNonArrayData::Vec4 => "float4",
                    ast::TypeSpecifierNonArrayData::Mat2 => "float2x2",
                    ast::TypeSpecifierNonArrayData::Mat3 => "float3x3",
                    ast::TypeSpecifierNonArrayData::Mat4 => "float4x4",
                    _ => "/* unknown constructor */",
                }
            }
            ast::FunIdentifierData::Expr(expr) => {
                // Handle function name expressions - typically simple identifiers
                match &expr.content {
                    ast::ExprData::Variable(identifier) => {
                        &identifier.content.0
                    }
                    _ => {
                        // Complex function expressions - for now return placeholder
                        return Ok("/* complex function call */".to_string());
                    }
                }
            }
        };

        // Map function name to HLSL equivalent
        let hlsl_func_name = self.map_function_name(func_name);

        // Translate arguments
        let mut arg_strings = Vec::new();
        for arg in args {
            arg_strings.push(self.translate_expression(&arg.content)?);
        }

        Ok(format!("{}({})", hlsl_func_name, arg_strings.join(", ")))
    }

    /// Translate unary operators
    fn translate_unary_op(&self, op: &ast::UnaryOpData) -> &'static str {
        match op {
            ast::UnaryOpData::Inc => "++",
            ast::UnaryOpData::Dec => "--",
            ast::UnaryOpData::Add => "+",
            ast::UnaryOpData::Minus => "-",
            ast::UnaryOpData::Not => "!",
            ast::UnaryOpData::Complement => "~",
        }
    }

    /// Translate binary operators
    fn translate_binary_op(&self, op: &ast::BinaryOpData) -> &'static str {
        match op {
            ast::BinaryOpData::Or => "||",
            ast::BinaryOpData::Xor => "^^",
            ast::BinaryOpData::And => "&&",
            ast::BinaryOpData::BitOr => "|",
            ast::BinaryOpData::BitXor => "^",
            ast::BinaryOpData::BitAnd => "&",
            ast::BinaryOpData::Equal => "==",
            ast::BinaryOpData::NonEqual => "!=",
            ast::BinaryOpData::Lt => "<",
            ast::BinaryOpData::Gt => ">",
            ast::BinaryOpData::Lte => "<=",
            ast::BinaryOpData::Gte => ">=",
            ast::BinaryOpData::LShift => "<<",
            ast::BinaryOpData::RShift => ">>",
            ast::BinaryOpData::Add => "+",
            ast::BinaryOpData::Sub => "-",
            ast::BinaryOpData::Mult => "*",
            ast::BinaryOpData::Div => "/",
            ast::BinaryOpData::Mod => "%",
        }
    }

    /// Translate assignment operators
    fn translate_assignment_op(&self, op: &ast::AssignmentOpData) -> &'static str {
        match op {
            ast::AssignmentOpData::Equal => "=",
            ast::AssignmentOpData::Mult => "*=",
            ast::AssignmentOpData::Div => "/=",
            ast::AssignmentOpData::Mod => "%=",
            ast::AssignmentOpData::Add => "+=",
            ast::AssignmentOpData::Sub => "-=",
            ast::AssignmentOpData::LShift => "<<=",
            ast::AssignmentOpData::RShift => ">>=",
            ast::AssignmentOpData::And => "&=",
            ast::AssignmentOpData::Xor => "^=",
            ast::AssignmentOpData::Or => "|=",
        }
    }

    /// Translate a declaration
    fn translate_declaration(&mut self, decl: &ast::DeclarationData) -> Result<(), String> {
        match decl {
            ast::DeclarationData::FunctionPrototype(_) => {
                // Handle function prototypes
                self.writeln("// Function prototype translation not implemented")?;
            }
            ast::DeclarationData::InitDeclaratorList(init_list) => {
                self.translate_init_declarator_list(&init_list.content)?;
            }
            ast::DeclarationData::Precision(_, _) => {
                // HLSL doesn't have precision qualifiers in the same way
                self.writeln("// Precision qualifier (GLSL-specific)")?;
            }
            ast::DeclarationData::Block(_) => {
                // Handle uniform/buffer blocks
                self.writeln("// Block declaration translation not implemented")?;
            }
            ast::DeclarationData::Invariant(_) => {
                // Handle invariant declarations
                self.writeln("// Invariant declaration (GLSL-specific)")?;
            }
        }
        Ok(())
    }

    /// Translate an init declarator list (variable declarations)
    fn translate_init_declarator_list(&mut self, init_list: &ast::InitDeclaratorListData) -> Result<(), String> {
        // Get the base type
        let base_type = self.glsl_type_to_hlsl(&init_list.head.ty.content.ty.content)?;
        
        // Handle qualifiers (uniform, in, out, etc.)
        let mut qualifiers = Vec::new();
        
        if let Some(qualifier) = &init_list.head.ty.content.qualifier {
            qualifiers.extend(self.translate_type_qualifiers(&qualifier.content));
        }
        
        // Format the declaration
        let mut decl_line = String::new();
        if !qualifiers.is_empty() {
            decl_line.push_str(&qualifiers.join(" "));
            decl_line.push(' ');
        }
        decl_line.push_str(&base_type);
        decl_line.push(' ');
        
        // Handle the declarator(s)
        let mut declarations = Vec::new();
        
        // First declarator
        if let Some(name) = &init_list.head.name {
            let mut var_decl = name.content.0.to_string();
            
            // Handle array specifier
            if let Some(array_spec) = &init_list.head.array_specifier {
                for dimension in &array_spec.content.dimensions {
                    match &dimension.content {
                        ast::ArraySpecifierDimensionData::Unsized => {
                            var_decl.push_str("[]");
                        }
                        ast::ArraySpecifierDimensionData::ExplicitlySized(expr) => {
                            let size_expr = self.translate_expression(&expr.content)?;
                            var_decl.push_str(&format!("[{}]", size_expr));
                        }
                    }
                }
            }
            
            // Handle initializer
            if let Some(initializer) = &init_list.head.initializer {
                let init_expr = self.translate_initializer(&initializer.content)?;
                var_decl.push_str(&format!(" = {}", init_expr));
            }
            
            declarations.push(var_decl);
        }
        
        // Additional declarators
        for declarator in &init_list.tail {
            let mut var_decl = declarator.content.ident.content.ident.content.0.to_string();
            
            // Handle initializer for additional declarators
            if let Some(initializer) = &declarator.content.initializer {
                let init_expr = self.translate_initializer(&initializer.content)?;
                var_decl.push_str(&format!(" = {}", init_expr));
            }
            
            declarations.push(var_decl);
        }
        
        decl_line.push_str(&declarations.join(", "));
        decl_line.push(';');
        
        self.writeln(&decl_line)?;
        Ok(())
    }

    /// Translate type qualifiers
    fn translate_type_qualifiers(&self, qualifier: &ast::TypeQualifierData) -> Vec<String> {
        let mut result = Vec::new();
        
        for qual_spec in &qualifier.qualifiers {
            match &qual_spec.content {
                ast::TypeQualifierSpecData::Storage(storage) => {
                    match &storage.content {
                        ast::StorageQualifierData::Const => result.push("const".to_string()),
                        ast::StorageQualifierData::In => result.push("/* in */".to_string()),
                        ast::StorageQualifierData::Out => result.push("/* out */".to_string()),
                        ast::StorageQualifierData::InOut => result.push("inout".to_string()),
                        ast::StorageQualifierData::Uniform => result.push("/* uniform */".to_string()),
                        ast::StorageQualifierData::Shared => result.push("groupshared".to_string()),
                        _ => result.push("/* storage qualifier */".to_string()),
                    }
                }
                ast::TypeQualifierSpecData::Layout(_) => {
                    result.push("/* layout qualifier */".to_string());
                }
                ast::TypeQualifierSpecData::Precision(_) => {
                    // HLSL doesn't have precision qualifiers in the same way
                }
                ast::TypeQualifierSpecData::Interpolation(interp) => {
                    match &interp.content {
                        ast::InterpolationQualifierData::Smooth => result.push("/* smooth */".to_string()),
                        ast::InterpolationQualifierData::Flat => result.push("nointerpolation".to_string()),
                        ast::InterpolationQualifierData::NoPerspective => result.push("noperspective".to_string()),
                    }
                }
                ast::TypeQualifierSpecData::Invariant => {
                    result.push("/* invariant */".to_string());
                }
                ast::TypeQualifierSpecData::Precise => {
                    result.push("precise".to_string());
                }
            }
        }
        
        result
    }

    /// Translate an initializer
    fn translate_initializer(&mut self, init: &ast::InitializerData) -> Result<String, String> {
        match init {
            ast::InitializerData::Simple(expr) => {
                self.translate_expression(&expr.content)
            }
            ast::InitializerData::List(_init_list) => {
                // Simplified handling of initializer lists
                Ok("{ /* initializer list */ }".to_string())
            }
        }
    }

    /// Convert GLSL type specifier to HLSL type string  
    fn glsl_type_to_hlsl(&self, type_spec: &ast::TypeSpecifierData) -> Result<String, String> {
        match &type_spec.ty.content {
            ast::TypeSpecifierNonArrayData::Void => Ok("void".to_string()),
            ast::TypeSpecifierNonArrayData::Bool => Ok("bool".to_string()),
            ast::TypeSpecifierNonArrayData::Int => Ok("int".to_string()),
            ast::TypeSpecifierNonArrayData::UInt => Ok("uint".to_string()),
            ast::TypeSpecifierNonArrayData::Float => Ok("float".to_string()),
            ast::TypeSpecifierNonArrayData::Double => Ok("double".to_string()),
            ast::TypeSpecifierNonArrayData::Vec2 => Ok("float2".to_string()),
            ast::TypeSpecifierNonArrayData::Vec3 => Ok("float3".to_string()),
            ast::TypeSpecifierNonArrayData::Vec4 => Ok("float4".to_string()),
            ast::TypeSpecifierNonArrayData::BVec2 => Ok("bool2".to_string()),
            ast::TypeSpecifierNonArrayData::BVec3 => Ok("bool3".to_string()),
            ast::TypeSpecifierNonArrayData::BVec4 => Ok("bool4".to_string()),
            ast::TypeSpecifierNonArrayData::IVec2 => Ok("int2".to_string()),
            ast::TypeSpecifierNonArrayData::IVec3 => Ok("int3".to_string()),
            ast::TypeSpecifierNonArrayData::IVec4 => Ok("int4".to_string()),
            ast::TypeSpecifierNonArrayData::UVec2 => Ok("uint2".to_string()),
            ast::TypeSpecifierNonArrayData::UVec3 => Ok("uint3".to_string()),
            ast::TypeSpecifierNonArrayData::UVec4 => Ok("uint4".to_string()),
            ast::TypeSpecifierNonArrayData::Mat2 => Ok("float2x2".to_string()),
            ast::TypeSpecifierNonArrayData::Mat3 => Ok("float3x3".to_string()),
            ast::TypeSpecifierNonArrayData::Mat4 => Ok("float4x4".to_string()),
            ast::TypeSpecifierNonArrayData::Mat23 => Ok("float2x3".to_string()),
            ast::TypeSpecifierNonArrayData::Mat24 => Ok("float2x4".to_string()),
            ast::TypeSpecifierNonArrayData::Mat32 => Ok("float3x2".to_string()),
            ast::TypeSpecifierNonArrayData::Mat34 => Ok("float3x4".to_string()),
            ast::TypeSpecifierNonArrayData::Mat42 => Ok("float4x2".to_string()),
            ast::TypeSpecifierNonArrayData::Mat43 => Ok("float4x3".to_string()),
            ast::TypeSpecifierNonArrayData::Sampler2D => Ok("Texture2D".to_string()),
            ast::TypeSpecifierNonArrayData::Sampler3D => Ok("Texture3D".to_string()),
            ast::TypeSpecifierNonArrayData::SamplerCube => Ok("TextureCube".to_string()),
            ast::TypeSpecifierNonArrayData::Sampler2DArray => Ok("Texture2DArray".to_string()),
            _ => Err(format!("Unsupported GLSL type: {:?}", type_spec)),
        }
    }

    /// Write a line with proper indentation
    fn writeln(&mut self, line: &str) -> Result<(), String> {
        for _ in 0..self.indent_level {
            self.output.push_str("    ");
        }
        self.output.push_str(line);
        self.output.push('\n');
        Ok(())
    }

    /// Map GLSL function name to HLSL equivalent
    fn map_function_name(&self, glsl_name: &str) -> String {
        self.function_mappings
            .get(glsl_name)
            .cloned()
            .unwrap_or_else(|| glsl_name.to_string())
    }

    /// Map GLSL built-in variable to HLSL semantic
    fn map_builtin_variable(&self, glsl_var: &str) -> Option<&GLSLToHLSLMapping> {
        self.builtin_mappings
            .iter()
            .find(|mapping| mapping.glsl_var == glsl_var)
    }

    /// Clear the output buffer
    pub fn clear(&mut self) {
        self.output.clear();
        self.indent_level = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_mapping() {
        let translator = HLSLTranslator::new();
        assert_eq!(translator.map_function_name("fract"), "frac");
        assert_eq!(translator.map_function_name("mix"), "lerp");
        assert_eq!(translator.map_function_name("inversesqrt"), "rsqrt");
        assert_eq!(translator.map_function_name("unknown_function"), "unknown_function");
    }

    #[test]
    fn test_builtin_variable_mapping() {
        let translator = HLSLTranslator::new();
        let mapping = translator.map_builtin_variable("gl_Position");
        assert!(mapping.is_some());
        assert_eq!(mapping.unwrap().hlsl_semantic, "SV_Position");
    }

    #[test]
    fn test_hlsl_type_display() {
        assert_eq!(format!("{}", HLSLType::Float3), "float3");
        assert_eq!(format!("{}", HLSLType::Float4x4), "float4x4");
        assert_eq!(format!("{}", HLSLType::Texture2D), "Texture2D");
    }
}
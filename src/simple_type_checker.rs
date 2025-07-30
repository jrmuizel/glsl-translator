use glsl_lang::ast::{self, Node};
use std::collections::HashMap;
use std::fmt;

/// Represents GLSL data types
#[derive(Debug, Clone, PartialEq)]
pub enum GLSLType {
    // Basic scalar types
    Void,
    Bool,
    Int,
    UInt,
    Float,
    Double,
    
    // Vector types
    Vec2,
    Vec3,
    Vec4,
    BVec2,
    BVec3,
    BVec4,
    IVec2,
    IVec3,
    IVec4,
    UVec2,
    UVec3,
    UVec4,
    DVec2,
    DVec3,
    DVec4,
    
    // Matrix types
    Mat2,
    Mat3,
    Mat4,
    
    // Array type
    Array(Box<GLSLType>, Option<usize>),
    
    // Struct type
    Struct(String, Vec<(String, GLSLType)>),
    
    // Function type
    Function(Box<GLSLType>, Vec<GLSLType>),
    
    // Unknown type for error handling
    Unknown,
}

impl fmt::Display for GLSLType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GLSLType::Void => write!(f, "void"),
            GLSLType::Bool => write!(f, "bool"),
            GLSLType::Int => write!(f, "int"),
            GLSLType::UInt => write!(f, "uint"),
            GLSLType::Float => write!(f, "float"),
            GLSLType::Double => write!(f, "double"),
            GLSLType::Vec2 => write!(f, "vec2"),
            GLSLType::Vec3 => write!(f, "vec3"),
            GLSLType::Vec4 => write!(f, "vec4"),
            GLSLType::BVec2 => write!(f, "bvec2"),
            GLSLType::BVec3 => write!(f, "bvec3"),
            GLSLType::BVec4 => write!(f, "bvec4"),
            GLSLType::IVec2 => write!(f, "ivec2"),
            GLSLType::IVec3 => write!(f, "ivec3"),
            GLSLType::IVec4 => write!(f, "ivec4"),
            GLSLType::UVec2 => write!(f, "uvec2"),
            GLSLType::UVec3 => write!(f, "uvec3"),
            GLSLType::UVec4 => write!(f, "uvec4"),
            GLSLType::DVec2 => write!(f, "dvec2"),
            GLSLType::DVec3 => write!(f, "dvec3"),
            GLSLType::DVec4 => write!(f, "dvec4"),
            GLSLType::Mat2 => write!(f, "mat2"),
            GLSLType::Mat3 => write!(f, "mat3"),
            GLSLType::Mat4 => write!(f, "mat4"),
            GLSLType::Array(ty, Some(size)) => write!(f, "{}[{}]", ty, size),
            GLSLType::Array(ty, None) => write!(f, "{}[]", ty),
            GLSLType::Struct(name, _) => write!(f, "struct {}", name),
            GLSLType::Function(ret, params) => {
                write!(f, "function(")?;
                for (i, param) in params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", param)?;
                }
                write!(f, ") -> {}", ret)
            },
            GLSLType::Unknown => write!(f, "<unknown>"),
        }
    }
}

impl GLSLType {
    /// Check if this type is a scalar type
    pub fn is_scalar(&self) -> bool {
        matches!(self, 
            GLSLType::Bool | GLSLType::Int | GLSLType::UInt | 
            GLSLType::Float | GLSLType::Double
        )
    }
    
    /// Check if this type is a vector type
    pub fn is_vector(&self) -> bool {
        matches!(self,
            GLSLType::Vec2 | GLSLType::Vec3 | GLSLType::Vec4 |
            GLSLType::BVec2 | GLSLType::BVec3 | GLSLType::BVec4 |
            GLSLType::IVec2 | GLSLType::IVec3 | GLSLType::IVec4 |
            GLSLType::UVec2 | GLSLType::UVec3 | GLSLType::UVec4 |
            GLSLType::DVec2 | GLSLType::DVec3 | GLSLType::DVec4
        )
    }
    
    /// Check if this type is a matrix type
    pub fn is_matrix(&self) -> bool {
        matches!(self, GLSLType::Mat2 | GLSLType::Mat3 | GLSLType::Mat4)
    }
    
    /// Check if this type is numeric (can be used in arithmetic operations)
    pub fn is_numeric(&self) -> bool {
        self.is_scalar() || self.is_vector() || self.is_matrix()
    }
    
    /// Get the component count for vectors and matrices
    pub fn component_count(&self) -> Option<usize> {
        match self {
            GLSLType::Vec2 | GLSLType::BVec2 | GLSLType::IVec2 | GLSLType::UVec2 | GLSLType::DVec2 => Some(2),
            GLSLType::Vec3 | GLSLType::BVec3 | GLSLType::IVec3 | GLSLType::UVec3 | GLSLType::DVec3 => Some(3),
            GLSLType::Vec4 | GLSLType::BVec4 | GLSLType::IVec4 | GLSLType::UVec4 | GLSLType::DVec4 => Some(4),
            GLSLType::Mat2 => Some(4),
            GLSLType::Mat3 => Some(9),
            GLSLType::Mat4 => Some(16),
            _ => None,
        }
    }
    
    /// Get the base scalar type for vectors and matrices
    pub fn base_type(&self) -> Option<GLSLType> {
        match self {
            GLSLType::Vec2 | GLSLType::Vec3 | GLSLType::Vec4 => Some(GLSLType::Float),
            GLSLType::BVec2 | GLSLType::BVec3 | GLSLType::BVec4 => Some(GLSLType::Bool),
            GLSLType::IVec2 | GLSLType::IVec3 | GLSLType::IVec4 => Some(GLSLType::Int),
            GLSLType::UVec2 | GLSLType::UVec3 | GLSLType::UVec4 => Some(GLSLType::UInt),
            GLSLType::DVec2 | GLSLType::DVec3 | GLSLType::DVec4 => Some(GLSLType::Double),
            GLSLType::Mat2 | GLSLType::Mat3 | GLSLType::Mat4 => Some(GLSLType::Float),
            _ => None,
        }
    }
    
    /// Check if two types are compatible for assignment or comparison
    pub fn is_compatible_with(&self, other: &GLSLType) -> bool {
        if self == other {
            return true;
        }
        
        // Allow implicit conversions
        match (self, other) {
            // int to uint, int/uint to float, int/uint/float to double
            (GLSLType::UInt, GLSLType::Int) => true,
            (GLSLType::Float, GLSLType::Int) | (GLSLType::Float, GLSLType::UInt) => true,
            (GLSLType::Double, GLSLType::Int) | (GLSLType::Double, GLSLType::UInt) | (GLSLType::Double, GLSLType::Float) => true,
            
            // Vector conversions follow the same rules
            (GLSLType::UVec2, GLSLType::IVec2) | (GLSLType::UVec3, GLSLType::IVec3) | (GLSLType::UVec4, GLSLType::IVec4) => true,
            (GLSLType::Vec2, GLSLType::IVec2) | (GLSLType::Vec2, GLSLType::UVec2) => true,
            (GLSLType::Vec3, GLSLType::IVec3) | (GLSLType::Vec3, GLSLType::UVec3) => true,
            (GLSLType::Vec4, GLSLType::IVec4) | (GLSLType::Vec4, GLSLType::UVec4) => true,
            (GLSLType::DVec2, GLSLType::IVec2) | (GLSLType::DVec2, GLSLType::UVec2) | (GLSLType::DVec2, GLSLType::Vec2) => true,
            (GLSLType::DVec3, GLSLType::IVec3) | (GLSLType::DVec3, GLSLType::UVec3) | (GLSLType::DVec3, GLSLType::Vec3) => true,
            (GLSLType::DVec4, GLSLType::IVec4) | (GLSLType::DVec4, GLSLType::UVec4) | (GLSLType::DVec4, GLSLType::Vec4) => true,
            
            _ => false,
        }
    }
}

/// Type checking error
#[derive(Debug, Clone)]
pub struct TypeError {
    pub message: String,
    pub line: Option<usize>,
    pub column: Option<usize>,
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.line, self.column) {
            (Some(line), Some(col)) => write!(f, "Type error at {}:{}: {}", line, col, self.message),
            (Some(line), None) => write!(f, "Type error at line {}: {}", line, self.message),
            _ => write!(f, "Type error: {}", self.message),
        }
    }
}

impl std::error::Error for TypeError {}

/// Symbol table for variable and function declarations
#[derive(Debug, Clone)]
pub struct SymbolTable {
    pub scopes: Vec<HashMap<String, GLSLType>>,  // Made public
    functions: HashMap<String, GLSLType>,
}

impl SymbolTable {
    pub fn new() -> Self {
        let mut table = Self {
            scopes: vec![HashMap::new()], // Global scope
            functions: HashMap::new(),
        };
        
        // Add built-in functions
        table.add_builtin_functions();
        table
    }
    
    pub fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }
    
    pub fn exit_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }
    
    pub fn declare_variable(&mut self, name: String, ty: GLSLType) -> Result<(), String> {
        if let Some(current_scope) = self.scopes.last_mut() {
            if current_scope.contains_key(&name) {
                return Err(format!("Variable '{}' already declared in current scope", name));
            }
            current_scope.insert(name, ty);
            Ok(())
        } else {
            Err("No active scope".to_string())
        }
    }
    
    pub fn lookup_variable(&self, name: &str) -> Option<&GLSLType> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        None
    }
    
    pub fn declare_function(&mut self, name: String, ty: GLSLType) {
        self.functions.insert(name, ty);
    }
    
    pub fn lookup_function(&self, name: &str) -> Option<&GLSLType> {
        self.functions.get(name)
    }
    
    fn add_builtin_functions(&mut self) {
        // Built-in mathematical functions
        let float_funcs = vec![
            ("sin", GLSLType::Float),
            ("cos", GLSLType::Float),
            ("tan", GLSLType::Float),
            ("sqrt", GLSLType::Float),
            ("abs", GLSLType::Float),
            ("floor", GLSLType::Float),
            ("ceil", GLSLType::Float),
        ];
        
        for (name, return_type) in float_funcs {
            self.functions.insert(
                name.to_string(),
                GLSLType::Function(Box::new(return_type.clone()), vec![GLSLType::Float])
            );
        }
        
        // Vector-specific functions
        self.functions.insert("length".to_string(), 
            GLSLType::Function(Box::new(GLSLType::Float), vec![GLSLType::Vec3]));
        self.functions.insert("dot".to_string(), 
            GLSLType::Function(Box::new(GLSLType::Float), vec![GLSLType::Vec3, GLSLType::Vec3]));
        self.functions.insert("normalize".to_string(), 
            GLSLType::Function(Box::new(GLSLType::Vec3), vec![GLSLType::Vec3]));
        
        // Constructor functions
        self.functions.insert("vec3".to_string(), 
            GLSLType::Function(Box::new(GLSLType::Vec3), vec![GLSLType::Float, GLSLType::Float, GLSLType::Float]));
        self.functions.insert("vec4".to_string(), 
            GLSLType::Function(Box::new(GLSLType::Vec4), vec![GLSLType::Float, GLSLType::Float, GLSLType::Float, GLSLType::Float]));
    }
}

/// Simplified type checker that demonstrates the core concepts
pub struct SimpleTypeChecker {
    pub symbol_table: SymbolTable,
    pub errors: Vec<TypeError>,
}

impl SimpleTypeChecker {
    pub fn new() -> Self {
        Self {
            symbol_table: SymbolTable::new(),
            errors: Vec::new(),
        }
    }
    
    pub fn check_translation_unit(&mut self, unit: &ast::TranslationUnit) -> Result<(), Vec<TypeError>> {
        for external_decl in &unit.0 {
            self.check_external_declaration(external_decl);
        }
        
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors.clone())
        }
    }
    
    fn error(&mut self, message: String) {
        self.errors.push(TypeError { message, line: None, column: None });
    }
    
    fn check_external_declaration(&mut self, decl: &Node<ast::ExternalDeclarationData>) {
        match &decl.content {
            ast::ExternalDeclarationData::Declaration(_) => {
                // Skip declarations for now
            },
            ast::ExternalDeclarationData::FunctionDefinition(node) => {
                self.check_function_definition(&node.content);
            },
            ast::ExternalDeclarationData::Preprocessor(_) => {
                // Skip preprocessor directives
            },
        }
    }
    
    fn check_function_definition(&mut self, func_def: &ast::FunctionDefinitionData) {
        let return_type = self.get_type_from_type_specifier(&func_def.prototype.content.ty.content.ty.content);
        
        // Enter function scope
        self.symbol_table.enter_scope();
        
        // Register function
        let func_name = func_def.prototype.content.name.content.0.to_string();
        let func_type = GLSLType::Function(Box::new(return_type), vec![]); // Simplified
        self.symbol_table.declare_function(func_name, func_type);
        
        // Check function body
        self.check_compound_statement(&func_def.statement.content);
        
        // Exit function scope
        self.symbol_table.exit_scope();
    }
    
    fn check_compound_statement(&mut self, stmt: &ast::CompoundStatementData) {
        for statement in &stmt.statement_list {
            self.check_statement(&statement.content);
        }
    }
    
    fn check_statement(&mut self, stmt: &ast::StatementData) {
        match stmt {
            ast::StatementData::Compound(node) => {
                self.symbol_table.enter_scope();
                self.check_compound_statement(&node.content);
                self.symbol_table.exit_scope();
            },
            ast::StatementData::Declaration(node) => {
                self.check_declaration(&node.content);
            },
            ast::StatementData::Expression(expr_stmt) => {
                // Fixed: ExprStatementData contains an Option<Expr>
                if let Some(expr) = &expr_stmt.content.0 {
                    self.check_expression(&expr.content);
                }
            },
            _ => {
                // Handle other statement types - simplified for now
            }
        }
    }
    
    fn check_declaration(&mut self, decl: &ast::DeclarationData) {
        match decl {
            ast::DeclarationData::InitDeclaratorList(node) => {
                self.check_init_declarator_list(&node.content);
            },
            _ => {
                // Handle other declaration types
            }
        }
    }
    
    fn check_init_declarator_list(&mut self, list: &ast::InitDeclaratorListData) {
        let base_type = self.get_type_from_fully_specified_type(&list.head.content.ty.content);
        
        // Check the first declarator
        if let Some(name) = &list.head.content.name {
            let var_name = name.content.0.to_string();
            
            if let Err(msg) = self.symbol_table.declare_variable(var_name, base_type.clone()) {
                self.error(msg);
            }
            
            // Check initializer if present
            if let Some(initializer) = &list.head.content.initializer {
                let init_type = self.check_initializer(&initializer.content);
                if !base_type.is_compatible_with(&init_type) {
                    self.error(
                        format!("Cannot initialize variable of type '{}' with value of type '{}'", 
                            base_type, init_type)
                    );
                }
            }
        }
    }
    
    fn check_initializer(&mut self, init: &ast::InitializerData) -> GLSLType {
        match init {
            ast::InitializerData::Simple(node) => {
                self.check_expression(&node.content)
            },
            ast::InitializerData::List(_) => {
                // Simplified handling for list initializers
                GLSLType::Unknown
            },
        }
    }
    
    fn check_expression(&mut self, expr: &ast::ExprData) -> GLSLType {
        match expr {
            ast::ExprData::Variable(node) => {
                let name = &node.content.0;
                if let Some(ty) = self.symbol_table.lookup_variable(name) {
                    ty.clone()
                } else {
                    self.error(format!("Undefined variable '{}'", name));
                    GLSLType::Unknown
                }
            },
            ast::ExprData::IntConst(_) => GLSLType::Int,
            ast::ExprData::UIntConst(_) => GLSLType::UInt,
            ast::ExprData::BoolConst(_) => GLSLType::Bool,
            ast::ExprData::FloatConst(_) => GLSLType::Float,
            ast::ExprData::DoubleConst(_) => GLSLType::Double,
            
            ast::ExprData::FunCall(fun, args) => {
                self.check_function_call(fun, args)
            },
            
            ast::ExprData::Binary(op, left, right) => {
                let left_type = self.check_expression(&left.content);
                let right_type = self.check_expression(&right.content);
                self.check_binary_operator(&op.content, &left_type, &right_type)
            },
            
            ast::ExprData::Assignment(left, _op, right) => {
                let left_type = self.check_expression(&left.content);
                let right_type = self.check_expression(&right.content);
                
                if !left_type.is_compatible_with(&right_type) {
                    self.error(format!("Cannot assign '{}' to '{}'", right_type, left_type));
                }
                
                left_type
            },
            
            _ => {
                // Handle other expression types
                GLSLType::Unknown
            }
        }
    }
    
    fn check_function_call(&mut self, fun: &ast::FunIdentifier, _args: &[ast::Expr]) -> GLSLType {
        match &fun.content {
            ast::FunIdentifierData::TypeSpecifier(type_spec) => {
                // Constructor call - get the type from the type specifier
                self.get_type_from_type_specifier(&type_spec.content)
            },
            ast::FunIdentifierData::Expr(_expr) => {
                // Function call through expression - simplified handling
                GLSLType::Unknown
            },
        }
    }
    
    fn check_binary_operator(&mut self, op: &ast::BinaryOpData, left: &GLSLType, right: &GLSLType) -> GLSLType {
        use ast::BinaryOpData::*;
        
        match op {
            Or | Xor | And => {
                if !matches!(left, GLSLType::Bool) || !matches!(right, GLSLType::Bool) {
                    self.error("Logical operators require boolean operands".to_string());
                }
                GLSLType::Bool
            },
            
            Equal | NonEqual => {
                if !left.is_compatible_with(right) && !right.is_compatible_with(left) {
                    self.error(format!("Cannot compare types '{}' and '{}'", left, right));
                }
                GLSLType::Bool
            },
            
            Lt | Lte | Gt | Gte => {
                if !left.is_numeric() || !right.is_numeric() {
                    self.error("Comparison operators require numeric operands".to_string());
                }
                GLSLType::Bool
            },
            
            Add | Sub | Mult | Div => {
                if !left.is_numeric() || !right.is_numeric() {
                    self.error("Arithmetic operators require numeric operands".to_string());
                    return GLSLType::Unknown;
                }
                
                // Simplified type promotion
                match (left, right) {
                    (l, r) if l == r => l.clone(),
                    (GLSLType::Float, _) | (_, GLSLType::Float) => GLSLType::Float,
                    (GLSLType::Int, _) | (_, GLSLType::Int) => GLSLType::Int,
                    _ => left.clone(),
                }
            },
            
            _ => {
                // Handle other operators
                GLSLType::Unknown
            }
        }
    }
    
    fn get_type_from_fully_specified_type(&self, spec: &ast::FullySpecifiedTypeData) -> GLSLType {
        self.get_type_from_type_specifier(&spec.ty.content)
    }
    
    fn get_type_from_type_specifier(&self, spec: &ast::TypeSpecifierData) -> GLSLType {
        match &spec.ty.content {
            ast::TypeSpecifierNonArrayData::Void => GLSLType::Void,
            ast::TypeSpecifierNonArrayData::Bool => GLSLType::Bool,
            ast::TypeSpecifierNonArrayData::Int => GLSLType::Int,
            ast::TypeSpecifierNonArrayData::UInt => GLSLType::UInt,
            ast::TypeSpecifierNonArrayData::Float => GLSLType::Float,
            ast::TypeSpecifierNonArrayData::Double => GLSLType::Double,
            ast::TypeSpecifierNonArrayData::Vec2 => GLSLType::Vec2,
            ast::TypeSpecifierNonArrayData::Vec3 => GLSLType::Vec3,
            ast::TypeSpecifierNonArrayData::Vec4 => GLSLType::Vec4,
            ast::TypeSpecifierNonArrayData::Mat2 => GLSLType::Mat2,
            ast::TypeSpecifierNonArrayData::Mat3 => GLSLType::Mat3,
            ast::TypeSpecifierNonArrayData::Mat4 => GLSLType::Mat4,
            _ => GLSLType::Unknown,
        }
    }
}
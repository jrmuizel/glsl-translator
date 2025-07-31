use glsl_lang::ast::{self, Node};
use std::collections::HashMap;
use std::fmt;

/// Represents GLSL data types
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)] // Many variants are used in tests
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
    Mat2x3,
    Mat2x4,
    Mat3x2,
    Mat3x4,
    Mat4x2,
    Mat4x3,
    DMat2,
    DMat3,
    DMat4,

    // Sampler types
    Sampler1D,
    Sampler2D,
    Sampler3D,
    SamplerCube,
    Sampler1DShadow,
    Sampler2DShadow,
    SamplerCubeShadow,
    Sampler1DArray,
    Sampler2DArray,
    Sampler1DArrayShadow,
    Sampler2DArrayShadow,
    ISampler1D,
    ISampler2D,
    ISampler3D,
    ISamplerCube,
    ISampler1DArray,
    ISampler2DArray,
    USampler1D,
    USampler2D,
    USampler3D,
    USamplerCube,
    USampler1DArray,
    USampler2DArray,

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
            GLSLType::Mat2x3 => write!(f, "mat2x3"),
            GLSLType::Mat2x4 => write!(f, "mat2x4"),
            GLSLType::Mat3x2 => write!(f, "mat3x2"),
            GLSLType::Mat3x4 => write!(f, "mat3x4"),
            GLSLType::Mat4x2 => write!(f, "mat4x2"),
            GLSLType::Mat4x3 => write!(f, "mat4x3"),
            GLSLType::DMat2 => write!(f, "dmat2"),
            GLSLType::DMat3 => write!(f, "dmat3"),
            GLSLType::DMat4 => write!(f, "dmat4"),
            GLSLType::Sampler1D => write!(f, "sampler1D"),
            GLSLType::Sampler2D => write!(f, "sampler2D"),
            GLSLType::Sampler3D => write!(f, "sampler3D"),
            GLSLType::SamplerCube => write!(f, "samplerCube"),
            GLSLType::Sampler1DShadow => write!(f, "sampler1DShadow"),
            GLSLType::Sampler2DShadow => write!(f, "sampler2DShadow"),
            GLSLType::SamplerCubeShadow => write!(f, "samplerCubeShadow"),
            GLSLType::Sampler1DArray => write!(f, "sampler1DArray"),
            GLSLType::Sampler2DArray => write!(f, "sampler2DArray"),
            GLSLType::Sampler1DArrayShadow => write!(f, "sampler1DArrayShadow"),
            GLSLType::Sampler2DArrayShadow => write!(f, "sampler2DArrayShadow"),
            GLSLType::ISampler1D => write!(f, "isampler1D"),
            GLSLType::ISampler2D => write!(f, "isampler2D"),
            GLSLType::ISampler3D => write!(f, "isampler3D"),
            GLSLType::ISamplerCube => write!(f, "isamplerCube"),
            GLSLType::ISampler1DArray => write!(f, "isampler1DArray"),
            GLSLType::ISampler2DArray => write!(f, "isampler2DArray"),
            GLSLType::USampler1D => write!(f, "usampler1D"),
            GLSLType::USampler2D => write!(f, "usampler2D"),
            GLSLType::USampler3D => write!(f, "usampler3D"),
            GLSLType::USamplerCube => write!(f, "usamplerCube"),
            GLSLType::USampler1DArray => write!(f, "usampler1DArray"),
            GLSLType::USampler2DArray => write!(f, "usampler2DArray"),
            GLSLType::Array(ty, Some(size)) => write!(f, "{ty}[{size}]"),
            GLSLType::Array(ty, None) => write!(f, "{ty}[]"),
            GLSLType::Struct(name, _) => write!(f, "struct {name}"),
            GLSLType::Function(ret, params) => {
                write!(f, "function(")?;
                for (i, param) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{param}")?;
                }
                write!(f, ") -> {ret}")
            }
            GLSLType::Unknown => write!(f, "<unknown>"),
        }
    }
}

impl GLSLType {
    /// Check if this type is a scalar type
    #[must_use]
    pub fn is_scalar(&self) -> bool {
        matches!(
            self,
            GLSLType::Bool | GLSLType::Int | GLSLType::UInt | GLSLType::Float | GLSLType::Double
        )
    }

    /// Check if this type is a vector type
    #[must_use]
    pub fn is_vector(&self) -> bool {
        matches!(
            self,
            GLSLType::Vec2
                | GLSLType::Vec3
                | GLSLType::Vec4
                | GLSLType::BVec2
                | GLSLType::BVec3
                | GLSLType::BVec4
                | GLSLType::IVec2
                | GLSLType::IVec3
                | GLSLType::IVec4
                | GLSLType::UVec2
                | GLSLType::UVec3
                | GLSLType::UVec4
                | GLSLType::DVec2
                | GLSLType::DVec3
                | GLSLType::DVec4
        )
    }

    /// Check if this type is a matrix type
    #[must_use]
    pub fn is_matrix(&self) -> bool {
        matches!(
            self,
            GLSLType::Mat2
                | GLSLType::Mat3
                | GLSLType::Mat4
                | GLSLType::Mat2x3
                | GLSLType::Mat2x4
                | GLSLType::Mat3x2
                | GLSLType::Mat3x4
                | GLSLType::Mat4x2
                | GLSLType::Mat4x3
                | GLSLType::DMat2
                | GLSLType::DMat3
                | GLSLType::DMat4
        )
    }

    /// Check if this type is numeric (can be used in arithmetic operations)
    #[must_use]
    pub fn is_numeric(&self) -> bool {
        match self {
            // Bool types (scalar and vector) are not numeric
            GLSLType::Bool | GLSLType::BVec2 | GLSLType::BVec3 | GLSLType::BVec4 => false,
            // All other scalar types, vector types, and matrix types are numeric
            _ => self.is_scalar() || self.is_vector() || self.is_matrix(),
        }
    }

    /// Get the component count for vectors and matrices
    #[allow(dead_code)] // Used in tests
    #[must_use]
    pub fn component_count(&self) -> Option<usize> {
        match self {
            GLSLType::Vec2
            | GLSLType::BVec2
            | GLSLType::IVec2
            | GLSLType::UVec2
            | GLSLType::DVec2 => Some(2),
            GLSLType::Vec3
            | GLSLType::BVec3
            | GLSLType::IVec3
            | GLSLType::UVec3
            | GLSLType::DVec3 => Some(3),
            GLSLType::Vec4
            | GLSLType::BVec4
            | GLSLType::IVec4
            | GLSLType::UVec4
            | GLSLType::DVec4
            | GLSLType::Mat2 => Some(4),
            GLSLType::Mat3 => Some(9),
            GLSLType::Mat4 => Some(16),
            _ => None,
        }
    }

    /// Get the base scalar type for vectors and matrices
    #[allow(dead_code)] // Used in tests
    #[must_use]
    pub fn base_type(&self) -> Option<GLSLType> {
        match self {
            GLSLType::Vec2 | GLSLType::Vec3 | GLSLType::Vec4 | GLSLType::Mat2 | GLSLType::Mat3 | GLSLType::Mat4 => Some(GLSLType::Float),
            GLSLType::BVec2 | GLSLType::BVec3 | GLSLType::BVec4 => Some(GLSLType::Bool),
            GLSLType::IVec2 | GLSLType::IVec3 | GLSLType::IVec4 => Some(GLSLType::Int),
            GLSLType::UVec2 | GLSLType::UVec3 | GLSLType::UVec4 => Some(GLSLType::UInt),
            GLSLType::DVec2 | GLSLType::DVec3 | GLSLType::DVec4 => Some(GLSLType::Double),
            _ => None,
        }
    }

    /// Check if this type can be constructed from the given argument types
    #[must_use]
    pub fn can_construct_from(&self, args: &[GLSLType]) -> bool {
        match self {
            // Vector constructors
            GLSLType::Vec2 => {
                match args.len() {
                    1 => args[0].is_scalar() && args[0].is_numeric(),
                    2 => args.iter().all(|t| t.is_scalar() && t.is_numeric()),
                    _ => false,
                }
            }
            GLSLType::Vec3 => {
                match args.len() {
                    1 => args[0].is_scalar() && args[0].is_numeric(),
                    3 => args.iter().all(|t| t.is_scalar() && t.is_numeric()),
                    _ => false,
                }
            }
            GLSLType::Vec4 => {
                match args.len() {
                    1 => args[0].is_scalar() && args[0].is_numeric(),
                    4 => args.iter().all(|t| t.is_scalar() && t.is_numeric()),
                    _ => false,
                }
            }
            // Matrix constructors
            GLSLType::Mat2 => {
                match args.len() {
                    1 => args[0].is_scalar() && args[0].is_numeric(),
                    4 => args.iter().all(|t| t.is_scalar() && t.is_numeric()),
                    _ => false,
                }
            }
            GLSLType::Mat3 => {
                match args.len() {
                    1 => args[0].is_scalar() && args[0].is_numeric(),
                    9 => args.iter().all(|t| t.is_scalar() && t.is_numeric()),
                    _ => false,
                }
            }
            GLSLType::Mat4 => {
                match args.len() {
                    1 => args[0].is_scalar() && args[0].is_numeric(),
                    16 => args.iter().all(|t| t.is_scalar() && t.is_numeric()),
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// Check if two types are compatible for assignment or comparison
    #[must_use]
    pub fn is_compatible_with(&self, other: &GLSLType) -> bool {
        if self == other {
            return true;
        }

        // Allow implicit conversions
        #[allow(clippy::unnested_or_patterns)]
        match (self, other) {
            // Scalar type conversions: int to uint, int/uint to float, int/uint/float to double
            (GLSLType::UInt, GLSLType::Int)
            | (GLSLType::Float, GLSLType::Int) | (GLSLType::Float, GLSLType::UInt)
            | (GLSLType::Double, GLSLType::Int)
            | (GLSLType::Double, GLSLType::UInt)
            | (GLSLType::Double, GLSLType::Float)
            // Vector conversions follow the same rules
            | (GLSLType::UVec2, GLSLType::IVec2)
            | (GLSLType::UVec3, GLSLType::IVec3)
            | (GLSLType::UVec4, GLSLType::IVec4)
            | (GLSLType::Vec2, GLSLType::IVec2) | (GLSLType::Vec2, GLSLType::UVec2)
            | (GLSLType::Vec3, GLSLType::IVec3) | (GLSLType::Vec3, GLSLType::UVec3)
            | (GLSLType::Vec4, GLSLType::IVec4) | (GLSLType::Vec4, GLSLType::UVec4)
            | (GLSLType::DVec2, GLSLType::IVec2)
            | (GLSLType::DVec2, GLSLType::UVec2)
            | (GLSLType::DVec2, GLSLType::Vec2)
            | (GLSLType::DVec3, GLSLType::IVec3)
            | (GLSLType::DVec3, GLSLType::UVec3)
            | (GLSLType::DVec3, GLSLType::Vec3)
            | (GLSLType::DVec4, GLSLType::IVec4)
            | (GLSLType::DVec4, GLSLType::UVec4)
            | (GLSLType::DVec4, GLSLType::Vec4) => true,

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
            (Some(line), Some(col)) => {
                write!(f, "Type error at {}:{}: {}", line, col, self.message)
            }
            (Some(line), None) => write!(f, "Type error at line {}: {}", line, self.message),
            _ => write!(f, "Type error: {}", self.message),
        }
    }
}

impl std::error::Error for TypeError {}

/// Symbol table for variable and function declarations
#[derive(Debug, Clone)]
pub struct SymbolTable {
    pub scopes: Vec<HashMap<String, GLSLType>>, // Made public
    functions: HashMap<String, GLSLType>,
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolTable {
    #[must_use]
    pub fn new() -> Self {
        let mut table = Self {
            scopes: vec![HashMap::new()], // Global scope
            functions: HashMap::new(),
        };

        // Add built-in functions
        table.add_builtin_functions();
        table.add_builtin_variables();
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

    /// Declares a variable in the current scope.
    /// 
    /// # Errors
    /// 
    /// Returns an error if a variable with the same name is already declared in the current scope.
    pub fn declare_variable(&mut self, name: String, ty: GLSLType) -> Result<(), String> {
        if let Some(current_scope) = self.scopes.last_mut() {
            if current_scope.contains_key(&name) {
                return Err(format!(
                    "Variable '{name}' already declared in current scope"
                ));
            }
            current_scope.insert(name, ty);
            Ok(())
        } else {
            Err("No active scope".to_string())
        }
    }

    #[must_use]
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

    #[allow(dead_code)] // Used in tests
    #[must_use]
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
                GLSLType::Function(Box::new(return_type.clone()), vec![GLSLType::Float]),
            );
        }

        // Vector-specific functions
        self.functions.insert(
            "length".to_string(),
            GLSLType::Function(Box::new(GLSLType::Float), vec![GLSLType::Vec3]),
        );
        self.functions.insert(
            "dot".to_string(),
            GLSLType::Function(
                Box::new(GLSLType::Float),
                vec![GLSLType::Vec3, GLSLType::Vec3],
            ),
        );
        self.functions.insert(
            "normalize".to_string(),
            GLSLType::Function(Box::new(GLSLType::Vec3), vec![GLSLType::Vec3]),
        );

        // Constructor functions
        self.functions.insert(
            "vec3".to_string(),
            GLSLType::Function(
                Box::new(GLSLType::Vec3),
                vec![GLSLType::Float, GLSLType::Float, GLSLType::Float],
            ),
        );
        self.functions.insert(
            "vec4".to_string(),
            GLSLType::Function(
                Box::new(GLSLType::Vec4),
                vec![
                    GLSLType::Float,
                    GLSLType::Float,
                    GLSLType::Float,
                    GLSLType::Float,
                ],
            ),
        );
    }

    fn add_builtin_variables(&mut self) {
        if let Some(global_scope) = self.scopes.first_mut() {
            // Vertex shader built-ins
            global_scope.insert("gl_Position".to_string(), GLSLType::Vec4);
            global_scope.insert("gl_PointSize".to_string(), GLSLType::Float);
            global_scope.insert("gl_VertexID".to_string(), GLSLType::Int);
            global_scope.insert("gl_InstanceID".to_string(), GLSLType::Int);

            // Fragment shader built-ins
            global_scope.insert("gl_FragColor".to_string(), GLSLType::Vec4);
            global_scope.insert("gl_FragData".to_string(), GLSLType::Array(Box::new(GLSLType::Vec4), None));
            global_scope.insert("gl_FragCoord".to_string(), GLSLType::Vec4);
            global_scope.insert("gl_FrontFacing".to_string(), GLSLType::Bool);
            global_scope.insert("gl_PointCoord".to_string(), GLSLType::Vec2);

            // Geometry shader built-ins
            global_scope.insert("gl_PrimitiveIDIn".to_string(), GLSLType::Int);
            global_scope.insert("gl_PrimitiveID".to_string(), GLSLType::Int);

            // Common built-ins
            global_scope.insert("gl_ClipDistance".to_string(), GLSLType::Array(Box::new(GLSLType::Float), None));
        }
    }
}

/// Simplified type checker that demonstrates the core concepts
pub struct SimpleTypeChecker {
    pub symbol_table: SymbolTable,
    pub errors: Vec<TypeError>,
}

impl Default for SimpleTypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleTypeChecker {
    #[must_use]
    pub fn new() -> Self {
        Self {
            symbol_table: SymbolTable::new(),
            errors: Vec::new(),
        }
    }

    /// Performs type checking on a translation unit.
    /// 
    /// # Errors
    /// 
    /// Returns a vector of type errors if any type checking violations are found.
    pub fn check_translation_unit(
        &mut self,
        unit: &ast::TranslationUnit,
    ) -> Result<(), Vec<TypeError>> {
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
        self.errors.push(TypeError {
            message,
            line: None,
            column: None,
        });
    }

    fn check_external_declaration(&mut self, decl: &Node<ast::ExternalDeclarationData>) {
        match &decl.content {
            ast::ExternalDeclarationData::Declaration(_) | ast::ExternalDeclarationData::Preprocessor(_) => {
                // Skip declarations and preprocessor directives for now
            }
            ast::ExternalDeclarationData::FunctionDefinition(node) => {
                self.check_function_definition(&node.content);
            }
        }
    }

    fn check_function_definition(&mut self, func_def: &ast::FunctionDefinitionData) {
        let return_type =
            Self::get_type_from_type_specifier(&func_def.prototype.content.ty.content.ty.content);

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
            }
            ast::StatementData::Declaration(node) => {
                self.check_declaration(&node.content);
            }
            ast::StatementData::Expression(expr_stmt) => {
                // Fixed: ExprStatementData contains an Option<Expr>
                if let Some(expr) = &expr_stmt.content.0 {
                    self.check_expression(&expr.content);
                }
            }
            ast::StatementData::Selection(selection) => {
                self.check_selection_statement(&selection.content);
            }
            ast::StatementData::Switch(switch) => {
                self.check_switch_statement(&switch.content);
            }
            ast::StatementData::CaseLabel(_) => {
                // Case labels are handled in switch statements
            }
            ast::StatementData::Iteration(iteration) => {
                self.check_iteration_statement(&iteration.content);
            }
            ast::StatementData::Jump(jump) => {
                self.check_jump_statement(&jump.content);
            }
            _ => {
                // Handle other statement types - simplified for now
            }
        }
    }

    fn check_selection_statement(&mut self, _selection: &ast::SelectionStatementData) {
        // Simplified selection statement checking
        // TODO: Properly handle selection statement structure
    }

    fn check_switch_statement(&mut self, _switch: &ast::SwitchStatementData) {
        // Simplified switch statement checking
        // TODO: Properly handle switch statement structure
    }

    fn check_iteration_statement(&mut self, _iteration: &ast::IterationStatementData) {
        // Simplified iteration statement checking  
        // TODO: Properly handle loop statement structures
    }

    fn check_jump_statement(&mut self, jump: &ast::JumpStatementData) {
        match jump {
            ast::JumpStatementData::Continue => {
                // Continue statements are valid in loops - context checking could be added
            }
            ast::JumpStatementData::Break => {
                // Break statements are valid in loops and switches - context checking could be added
            }
            ast::JumpStatementData::Return(expr) => {
                if let Some(return_expr) = expr {
                    let return_type = self.check_expression(&return_expr.content);
                    // TODO: Check against function return type
                    // For now, we just validate the expression
                    if matches!(return_type, GLSLType::Unknown) {
                        self.error("Invalid return expression".to_string());
                    }
                }
            }
            ast::JumpStatementData::Discard => {
                // Discard is valid in fragment shaders
            }
        }
    }

    fn check_declaration(&mut self, decl: &ast::DeclarationData) {
        if let ast::DeclarationData::InitDeclaratorList(node) = decl {
            self.check_init_declarator_list(&node.content);
        } else {
            // Handle other declaration types
        }
    }

    fn check_init_declarator_list(&mut self, list: &ast::InitDeclaratorListData) {
        let base_type = Self::get_type_from_fully_specified_type(&list.head.content.ty.content);

        // Check the first declarator
        if let Some(name) = &list.head.content.name {
            let var_name = name.content.0.to_string();

            if let Err(msg) = self
                .symbol_table
                .declare_variable(var_name, base_type.clone())
            {
                self.error(msg);
            }

            // Check initializer if present
            if let Some(initializer) = &list.head.content.initializer {
                let init_type = self.check_initializer(&initializer.content);
                if !base_type.is_compatible_with(&init_type) {
                    self.error(format!(
                        "Cannot initialize variable of type '{base_type}' with value of type '{init_type}'"
                    ));
                }
            }
        }
    }

    fn check_initializer(&mut self, init: &ast::InitializerData) -> GLSLType {
        match init {
            ast::InitializerData::Simple(node) => self.check_expression(&node.content),
            ast::InitializerData::List(_) => {
                // Simplified handling for list initializers
                GLSLType::Unknown
            }
        }
    }

    fn check_expression(&mut self, expr: &ast::ExprData) -> GLSLType {
        match expr {
            ast::ExprData::Variable(node) => {
                let name = &node.content.0;
                if let Some(ty) = self.symbol_table.lookup_variable(name) {
                    ty.clone()
                } else {
                    self.error(format!("Undefined variable '{name}'"));
                    GLSLType::Unknown
                }
            }
            ast::ExprData::IntConst(_) => GLSLType::Int,
            ast::ExprData::UIntConst(_) => GLSLType::UInt,
            ast::ExprData::BoolConst(_) => GLSLType::Bool,
            ast::ExprData::FloatConst(_) => GLSLType::Float,
            ast::ExprData::DoubleConst(_) => GLSLType::Double,

            ast::ExprData::FunCall(fun, args) => self.check_function_call(fun, args),

            ast::ExprData::Binary(op, left, right) => {
                let left_type = self.check_expression(&left.content);
                let right_type = self.check_expression(&right.content);
                self.check_binary_operator(&op.content, &left_type, &right_type)
            }

            ast::ExprData::Assignment(left, _op, right) => {
                let left_type = self.check_expression(&left.content);
                let right_type = self.check_expression(&right.content);

                if !left_type.is_compatible_with(&right_type) {
                    self.error(format!("Cannot assign '{right_type}' to '{left_type}'"));
                }

                left_type
            }

            ast::ExprData::Unary(op, expr) => {
                let expr_type = self.check_expression(&expr.content);
                self.check_unary_operator(&op.content, &expr_type)
            }

            ast::ExprData::PostInc(expr) | ast::ExprData::PostDec(expr) => {
                let expr_type = self.check_expression(&expr.content);
                if !expr_type.is_numeric() {
                    self.error("Increment/decrement operators require numeric operands".to_string());
                }
                expr_type
            }

            ast::ExprData::Comma(left, right) => {
                self.check_expression(&left.content);
                self.check_expression(&right.content)
            }

            ast::ExprData::Ternary(cond, true_expr, false_expr) => {
                let cond_type = self.check_expression(&cond.content);
                let true_type = self.check_expression(&true_expr.content);
                let false_type = self.check_expression(&false_expr.content);

                if !matches!(cond_type, GLSLType::Bool) {
                    self.error("Ternary condition must be boolean".to_string());
                }

                if true_type.is_compatible_with(&false_type) {
                    true_type
                } else if false_type.is_compatible_with(&true_type) {
                    false_type
                } else {
                    self.error(format!(
                        "Incompatible types in ternary operator: '{}' and '{}'",
                        true_type, false_type
                    ));
                    GLSLType::Unknown
                }
            }

            ast::ExprData::Bracket(expr, index) => {
                let expr_type = self.check_expression(&expr.content);
                let index_type = self.check_expression(&index.content);

                if !matches!(index_type, GLSLType::Int | GLSLType::UInt) {
                    self.error("Array index must be integer".to_string());
                }

                match expr_type {
                    GLSLType::Array(element_type, _) => (*element_type).clone(),
                    GLSLType::Vec2 | GLSLType::Vec3 | GLSLType::Vec4 => GLSLType::Float,
                    GLSLType::IVec2 | GLSLType::IVec3 | GLSLType::IVec4 => GLSLType::Int,
                    GLSLType::BVec2 | GLSLType::BVec3 | GLSLType::BVec4 => GLSLType::Bool,
                    GLSLType::UVec2 | GLSLType::UVec3 | GLSLType::UVec4 => GLSLType::UInt,
                    GLSLType::DVec2 | GLSLType::DVec3 | GLSLType::DVec4 => GLSLType::Double,
                    GLSLType::Mat2 | GLSLType::Mat3 | GLSLType::Mat4 => GLSLType::Vec4, // Simplified
                    _ => {
                        self.error(format!("Cannot index into type '{}'", expr_type));
                        GLSLType::Unknown
                    }
                }
            }

            ast::ExprData::Dot(expr, field) => {
                let expr_type = self.check_expression(&expr.content);
                self.check_field_access(&expr_type, &field.content.0)
            }

            _ => {
                // Handle other expression types that aren't implemented yet
                GLSLType::Unknown
            }
        }
    }

    fn check_function_call(&mut self, fun: &ast::FunIdentifier, args: &[ast::Expr]) -> GLSLType {
        // First, evaluate all argument types
        let arg_types: Vec<GLSLType> = args.iter()
            .map(|arg| self.check_expression(&arg.content))
            .collect();

        match &fun.content {
            ast::FunIdentifierData::TypeSpecifier(type_spec) => {
                // Constructor call - get the type from the type specifier
                let target_type = Self::get_type_from_type_specifier(&type_spec.content);
                
                // Check if the constructor arguments are valid
                if target_type.can_construct_from(&arg_types) {
                    target_type
                } else {
                    self.error(format!(
                        "Cannot construct '{}' with arguments ({})",
                        target_type,
                        arg_types.iter()
                            .map(|t| t.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                    GLSLType::Unknown
                }
            }
            ast::FunIdentifierData::Expr(expr) => {
                // Function call through expression
                if let ast::ExprData::Variable(var) = &expr.content {
                    let func_name = &var.content.0;
                    
                    // Check if it's a built-in function
                    if let Some(func_type) = self.symbol_table.lookup_function(func_name) {
                        if let GLSLType::Function(return_type, param_types) = func_type {
                            // Check argument compatibility (simplified - exact match for now)
                            if param_types.len() == arg_types.len() {
                                let mut compatible = true;
                                for (param_type, arg_type) in param_types.iter().zip(arg_types.iter()) {
                                    if !param_type.is_compatible_with(arg_type) && !arg_type.is_compatible_with(param_type) {
                                        compatible = false;
                                        break;
                                    }
                                }
                                
                                if compatible {
                                    (**return_type).clone()
                                } else {
                                    self.error(format!(
                                        "Function '{}' called with incompatible arguments", 
                                        func_name
                                    ));
                                    GLSLType::Unknown
                                }
                            } else {
                                self.error(format!(
                                    "Function '{}' expects {} arguments, got {}",
                                    func_name, param_types.len(), arg_types.len()
                                ));
                                GLSLType::Unknown
                            }
                        } else {
                            self.error(format!("'{}' is not a function", func_name));
                            GLSLType::Unknown
                        }
                    } else {
                        self.error(format!("Undefined function '{}'", func_name));
                        GLSLType::Unknown
                    }
                } else {
                    // Complex function expression - simplified handling
                    GLSLType::Unknown
                }
            }
        }
    }

    fn check_binary_operator(
        &mut self,
        op: &ast::BinaryOpData,
        left: &GLSLType,
        right: &GLSLType,
    ) -> GLSLType {
        use ast::BinaryOpData::{Add, And, Div, Equal, Gt, Gte, Lt, Lte, Mult, NonEqual, Or, Sub, Xor};

        match op {
            Or | Xor | And => {
                if !matches!(left, GLSLType::Bool) || !matches!(right, GLSLType::Bool) {
                    self.error("Logical operators require boolean operands".to_string());
                }
                GLSLType::Bool
            }

            Equal | NonEqual => {
                if !left.is_compatible_with(right) && !right.is_compatible_with(left) {
                    self.error(format!("Cannot compare types '{left}' and '{right}'"));
                }
                GLSLType::Bool
            }

            Lt | Lte | Gt | Gte => {
                if !left.is_numeric() || !right.is_numeric() {
                    self.error("Comparison operators require numeric operands".to_string());
                }
                GLSLType::Bool
            }

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
            }

            _ => {
                // Handle other operators
                GLSLType::Unknown
            }
        }
    }

    fn check_unary_operator(
        &mut self,
        op: &ast::UnaryOpData,
        expr: &GLSLType,
    ) -> GLSLType {
        use ast::UnaryOpData::{Add, Not, Inc, Dec};

        match op {
            Add => expr.clone(),
            ast::UnaryOpData::Minus => {
                if !expr.is_numeric() {
                    self.error("Unary minus requires numeric operand".to_string());
                    return GLSLType::Unknown;
                }
                expr.clone()
            }
            Not => {
                if !matches!(expr, GLSLType::Bool) {
                    self.error("Unary not requires boolean operand".to_string());
                    return GLSLType::Unknown;
                }
                GLSLType::Bool
            }
            Inc => {
                if !expr.is_numeric() {
                    self.error("Increment operator requires numeric operand".to_string());
                    return GLSLType::Unknown;
                }
                expr.clone()
            }
            Dec => {
                if !expr.is_numeric() {
                    self.error("Decrement operator requires numeric operand".to_string());
                    return GLSLType::Unknown;
                }
                expr.clone()
            }
            _ => expr.clone(),
        }
    }

    fn check_field_access(&mut self, expr: &GLSLType, field: &str) -> GLSLType {
        match expr {
            GLSLType::Struct(_name, fields) => {
                if let Some((_field_name, field_type)) = fields.iter().find(|(n, _)| n == field) {
                    field_type.clone()
                } else {
                    self.error(format!("Field '{}' not found in struct", field));
                    GLSLType::Unknown
                }
            }
            // Vector swizzling
            GLSLType::Vec2 | GLSLType::Vec3 | GLSLType::Vec4 => {
                self.check_vector_swizzle(expr, field)
            }
            GLSLType::IVec2 | GLSLType::IVec3 | GLSLType::IVec4 => {
                self.check_vector_swizzle(expr, field)
            }
            GLSLType::BVec2 | GLSLType::BVec3 | GLSLType::BVec4 => {
                self.check_vector_swizzle(expr, field)
            }
            GLSLType::UVec2 | GLSLType::UVec3 | GLSLType::UVec4 => {
                self.check_vector_swizzle(expr, field)
            }
            GLSLType::DVec2 | GLSLType::DVec3 | GLSLType::DVec4 => {
                self.check_vector_swizzle(expr, field)
            }
            _ => {
                self.error(format!("Cannot access field '{}' on type '{}'", field, expr));
                GLSLType::Unknown
            }
        }
    }

    fn check_vector_swizzle(&mut self, vec_type: &GLSLType, swizzle: &str) -> GLSLType {
        let base_type = vec_type.base_type().unwrap_or(GLSLType::Float);
        let vec_size = vec_type.component_count().unwrap_or(0);
        
        // Validate swizzle characters
        let valid_chars = if swizzle.chars().all(|c| "xyzw".contains(c)) {
            "xyzw"
        } else if swizzle.chars().all(|c| "rgba".contains(c)) {
            "rgba"
        } else if swizzle.chars().all(|c| "stpq".contains(c)) {
            "stpq"
        } else {
            self.error(format!("Invalid swizzle '{}'", swizzle));
            return GLSLType::Unknown;
        };

        // Check if all components are valid for the vector size
        for c in swizzle.chars() {
            let index = valid_chars.chars().position(|ch| ch == c).unwrap_or(4);
            if index >= vec_size {
                self.error(format!(
                    "Swizzle component '{}' is out of range for vector of size {}",
                    c, vec_size
                ));
                return GLSLType::Unknown;
            }
        }

        // Return appropriate vector type based on swizzle length
        match (swizzle.len(), base_type) {
            (1, GLSLType::Float) => GLSLType::Float,
            (1, GLSLType::Int) => GLSLType::Int,
            (1, GLSLType::Bool) => GLSLType::Bool,
            (1, GLSLType::UInt) => GLSLType::UInt,
            (1, GLSLType::Double) => GLSLType::Double,
            (2, GLSLType::Float) => GLSLType::Vec2,
            (2, GLSLType::Int) => GLSLType::IVec2,
            (2, GLSLType::Bool) => GLSLType::BVec2,
            (2, GLSLType::UInt) => GLSLType::UVec2,
            (2, GLSLType::Double) => GLSLType::DVec2,
            (3, GLSLType::Float) => GLSLType::Vec3,
            (3, GLSLType::Int) => GLSLType::IVec3,
            (3, GLSLType::Bool) => GLSLType::BVec3,
            (3, GLSLType::UInt) => GLSLType::UVec3,
            (3, GLSLType::Double) => GLSLType::DVec3,
            (4, GLSLType::Float) => GLSLType::Vec4,
            (4, GLSLType::Int) => GLSLType::IVec4,
            (4, GLSLType::Bool) => GLSLType::BVec4,
            (4, GLSLType::UInt) => GLSLType::UVec4,
            (4, GLSLType::Double) => GLSLType::DVec4,
            _ => {
                self.error(format!("Invalid swizzle length: {}", swizzle.len()));
                GLSLType::Unknown
            }
        }
    }

    fn get_type_from_fully_specified_type(spec: &ast::FullySpecifiedTypeData) -> GLSLType {
        Self::get_type_from_type_specifier(&spec.ty.content)
    }

    fn get_type_from_type_specifier(spec: &ast::TypeSpecifierData) -> GLSLType {
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
            ast::TypeSpecifierNonArrayData::BVec2 => GLSLType::BVec2,
            ast::TypeSpecifierNonArrayData::BVec3 => GLSLType::BVec3,
            ast::TypeSpecifierNonArrayData::BVec4 => GLSLType::BVec4,
            ast::TypeSpecifierNonArrayData::IVec2 => GLSLType::IVec2,
            ast::TypeSpecifierNonArrayData::IVec3 => GLSLType::IVec3,
            ast::TypeSpecifierNonArrayData::IVec4 => GLSLType::IVec4,
            ast::TypeSpecifierNonArrayData::UVec2 => GLSLType::UVec2,
            ast::TypeSpecifierNonArrayData::UVec3 => GLSLType::UVec3,
            ast::TypeSpecifierNonArrayData::UVec4 => GLSLType::UVec4,
            ast::TypeSpecifierNonArrayData::DVec2 => GLSLType::DVec2,
            ast::TypeSpecifierNonArrayData::DVec3 => GLSLType::DVec3,
            ast::TypeSpecifierNonArrayData::DVec4 => GLSLType::DVec4,
            ast::TypeSpecifierNonArrayData::Mat2 => GLSLType::Mat2,
            ast::TypeSpecifierNonArrayData::Mat3 => GLSLType::Mat3,
            ast::TypeSpecifierNonArrayData::Mat4 => GLSLType::Mat4,
            ast::TypeSpecifierNonArrayData::Mat23 => GLSLType::Mat2x3,
            ast::TypeSpecifierNonArrayData::Mat24 => GLSLType::Mat2x4,
            ast::TypeSpecifierNonArrayData::Mat32 => GLSLType::Mat3x2,
            ast::TypeSpecifierNonArrayData::Mat34 => GLSLType::Mat3x4,
            ast::TypeSpecifierNonArrayData::Mat42 => GLSLType::Mat4x2,
            ast::TypeSpecifierNonArrayData::Mat43 => GLSLType::Mat4x3,
            ast::TypeSpecifierNonArrayData::Sampler1D => GLSLType::Sampler1D,
            ast::TypeSpecifierNonArrayData::Sampler2D => GLSLType::Sampler2D,
            ast::TypeSpecifierNonArrayData::Sampler3D => GLSLType::Sampler3D,
            ast::TypeSpecifierNonArrayData::SamplerCube => GLSLType::SamplerCube,
            _ => GLSLType::Unknown,
        }
    }
}

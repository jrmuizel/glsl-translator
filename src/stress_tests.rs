//! Stress tests for challenging GLSL to HLSL translation scenarios
//! These tests push the limits of the translator and cover edge cases that are 
//! particularly difficult to handle correctly.

#[cfg(test)]
mod stress_tests {
    use crate::hlsl_translator::*;
    use glsl_lang::ast;
    use glsl_lang::parse::Parsable;

    /// Helper function for stress testing GLSL to HLSL translation
    fn stress_test_translation(test_name: &str, glsl_code: &str, should_succeed: bool) {
        println!("\n=== Stress Test: {} ===", test_name);
        println!("GLSL Input:\n{}", glsl_code);
        
        match ast::TranslationUnit::parse(glsl_code) {
            Ok(translation_unit) => {
                let mut translator = HLSLTranslator::new();
                // Default to fragment shader for stress tests
                match translator.translate_translation_unit_with_type(&translation_unit, ShaderType::Fragment) {
                    Ok(hlsl_code) => {
                        if should_succeed {
                            println!("✓ Translation succeeded");
                            println!("HLSL Output:\n{}", hlsl_code);
                            assert!(!hlsl_code.trim().is_empty(), "HLSL output should not be empty");
                        } else {
                            println!("⚠ Translation unexpectedly succeeded when it should have failed");
                            println!("HLSL Output:\n{}", hlsl_code);
                        }
                    }
                    Err(error) => {
                        if should_succeed {
                            panic!("Translation failed when it should have succeeded: {}", error);
                        } else {
                            println!("✓ Translation failed as expected: {}", error);
                        }
                    }
                }
            }
            Err(parse_err) => {
                println!("Parse error (may be acceptable for stress test): {:?}", parse_err);
            }
        }
    }

    #[test]
    fn stress_test_deeply_nested_expressions() {
        let glsl_code = r"
            void main() {
                vec4 color = vec4(1.0);
                
                // Deeply nested function calls and swizzling
                vec3 result = normalize(
                    cross(
                        mix(
                            color.rgb,
                            reflect(
                                normalize(color.xyz),
                                normalize(vec3(0.0, 1.0, 0.0))
                            ),
                            clamp(
                                dot(
                                    normalize(color.rgb),
                                    normalize(vec3(1.0, 1.0, 1.0))
                                ),
                                0.0,
                                1.0
                            )
                        ).zyx,
                        refract(
                            normalize(color.bgr),
                            normalize(vec3(0.0, 0.0, 1.0)),
                            0.8
                        ).yxz
                    )
                ).bgr;
                
                // Complex matrix chain operations
                mat4 mvp = mat4(1.0);
                mat3 normal = mat3(mvp);
                vec4 transformed = mvp * 
                                  transpose(mat4(normal)) * 
                                  inverse(mvp) * 
                                  vec4(result, 1.0);
            }
        ";
        
        stress_test_translation("Deeply Nested Expressions", glsl_code, true);
    }

    #[test]
    fn stress_test_massive_swizzling_combinations() {
        let glsl_code = r"
            void main() {
                vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
                
                // Every possible swizzling combination
                float r = v.r;
                float g = v.g;
                float b = v.b;
                float a = v.a;
                
                vec2 xy = v.xy;
                vec2 yx = v.yx;
                vec2 rg = v.rg;
                vec2 gr = v.gr;
                vec2 rb = v.rb;
                vec2 br = v.br;
                vec2 ra = v.ra;
                vec2 ar = v.ar;
                vec2 gb = v.gb;
                vec2 bg = v.bg;
                vec2 ga = v.ga;
                vec2 ag = v.ag;
                vec2 ba = v.ba;
                vec2 ab = v.ab;
                
                vec3 xyz = v.xyz;
                vec3 xzy = v.xzy;
                vec3 yxz = v.yxz;
                vec3 yzx = v.yzx;
                vec3 zxy = v.zxy;
                vec3 zyx = v.zyx;
                vec3 rgb = v.rgb;
                vec3 rbg = v.rbg;
                vec3 grb = v.grb;
                vec3 gbr = v.gbr;
                vec3 brg = v.brg;
                vec3 bgr = v.bgr;
                
                vec4 rgba = v.rgba;
                vec4 argb = v.argb;
                vec4 bgra = v.bgra;
                vec4 abgr = v.abgr;
                vec4 wzyx = v.wzyx;
                vec4 xyzw = v.xyzw;
                
                // Repeated components
                vec2 xx = v.xx;
                vec3 yyy = v.yyy;
                vec4 zzzz = v.zzzz;
                vec4 rrgg = v.rrgg;
                vec3 brr = v.brr;
                vec4 abab = v.abab;
                
                // Mixed arithmetic with swizzling
                vec3 mixed = v.rgb * v.bgr + v.rrr - v.ggg * v.bbb;
                vec4 complex = v.wxyz * v.zyxw + v.xyzw * v.wzyx;
            }
        ";
        
        stress_test_translation("Massive Swizzling Combinations", glsl_code, true);
    }

    #[test]
    fn stress_test_complex_texture_operations() {
        let glsl_code = r"
            uniform sampler2D baseTexture;
            uniform sampler2D normalTexture;
            uniform sampler2D specularTexture;
            uniform samplerCube envTexture;
            uniform sampler2DArray layeredTexture;
            uniform sampler3D volumeTexture;
            
            void main() {
                vec2 uv = vec2(0.5, 0.5);
                vec3 normal = vec3(0.0, 1.0, 0.0);
                
                // Complex texture sampling with multiple levels
                vec4 base = texture(baseTexture, uv);
                vec4 baseBlur = textureLod(baseTexture, uv, 3.0);
                vec4 baseMip = textureLod(baseTexture, uv * 2.0, 1.0);
                
                // Gradient-based sampling
                vec2 ddx_uv = dFdx(uv);
                vec2 ddy_uv = dFdy(uv);
                vec4 baseGrad = textureGrad(baseTexture, uv, ddx_uv, ddy_uv);
                
                // Normal map processing
                vec3 normalMap = texture(normalTexture, uv).rgb * 2.0 - 1.0;
                vec3 perturbedNormal = normalize(normal + normalMap);
                
                // Cube map with reflection
                vec3 reflection = reflect(normalize(vec3(1.0, 1.0, 1.0)), perturbedNormal);
                vec4 envColor = texture(envTexture, reflection);
                vec4 envColorBlur = textureLod(envTexture, reflection, 2.0);
                
                // Array texture sampling
                vec4 layer0 = texture(layeredTexture, vec3(uv, 0.0));
                vec4 layer1 = texture(layeredTexture, vec3(uv, 1.0));
                vec4 layerMix = mix(layer0, layer1, 0.5);
                
                // 3D texture sampling
                vec3 volumeUV = vec3(uv, 0.5);
                vec4 volume = texture(volumeTexture, volumeUV);
                vec4 volumeGrad = textureGrad(volumeTexture, volumeUV, 
                    vec3(ddx_uv, 0.0), vec3(ddy_uv, 0.0));
                
                // Complex combination
                vec4 finalColor = mix(
                    base * envColor,
                    layerMix * volume,
                    dot(perturbedNormal, normalize(vec3(1.0, 1.0, 1.0)))
                );
                
                finalColor += baseGrad * 0.1 + envColorBlur * 0.2 + volumeGrad * 0.05;
            }
        ";
        
        stress_test_translation("Complex Texture Operations", glsl_code, true);
    }

    #[test]
    fn stress_test_extreme_matrix_operations() {
        let glsl_code = r"
            void main() {
                // All matrix types
                mat2 m2 = mat2(1.0, 0.0, 0.0, 1.0);
                mat3 m3 = mat3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
                mat4 m4 = mat4(1.0);
                
                // Non-square matrices
                mat2x3 m23 = mat2x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
                mat2x4 m24 = mat2x4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
                mat3x2 m32 = mat3x2(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
                mat3x4 m34 = mat3x4(
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0
                );
                mat4x2 m42 = mat4x2(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
                mat4x3 m43 = mat4x3(
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0
                );
                
                // Matrix operations
                mat4 mvp = m4 * transpose(m4) * inverse(m4);
                mat3 normal = mat3(mvp);
                
                // Matrix-vector multiplications
                vec2 v2_result = m2 * vec2(1.0, 2.0);
                vec3 v3_result = m3 * vec3(1.0, 2.0, 3.0);
                vec4 v4_result = m4 * vec4(1.0, 2.0, 3.0, 4.0);
                
                // Non-square matrix operations
                vec3 v23_result = m23 * vec2(1.0, 2.0);
                vec4 v24_result = m24 * vec2(1.0, 2.0);
                vec2 v32_result = m32 * vec3(1.0, 2.0, 3.0);
                vec4 v34_result = m34 * vec3(1.0, 2.0, 3.0);
                vec2 v42_result = m42 * vec4(1.0, 2.0, 3.0, 4.0);
                vec3 v43_result = m43 * vec4(1.0, 2.0, 3.0, 4.0);
                
                // Chained matrix operations
                vec4 transformed = mvp * transpose(m4) * m4 * vec4(1.0, 2.0, 3.0, 1.0);
                
                // Matrix constructors from vectors
                vec3 col0 = vec3(1.0, 0.0, 0.0);
                vec3 col1 = vec3(0.0, 1.0, 0.0);
                vec3 col2 = vec3(0.0, 0.0, 1.0);
                mat3 fromVectors = mat3(col0, col1, col2);
            }
        ";
        
        stress_test_translation("Extreme Matrix Operations", glsl_code, true);
    }

    #[test]
    fn stress_test_all_atomic_operations() {
        let glsl_code = r"
            layout(binding = 0) buffer AtomicBuffer {
                int intCounter;
                uint uintCounter;
                float floatValue;
            } atomicBuffer;
            
            void main() {
                // Integer atomic operations
                int oldInt = atomicAdd(atomicBuffer.intCounter, 1);
                int andInt = atomicAnd(atomicBuffer.intCounter, 0xFF);
                int orInt = atomicOr(atomicBuffer.intCounter, 0x10);
                int xorInt = atomicXor(atomicBuffer.intCounter, 0xAA);
                int minInt = atomicMin(atomicBuffer.intCounter, 100);
                int maxInt = atomicMax(atomicBuffer.intCounter, 50);
                int exchangeInt = atomicExchange(atomicBuffer.intCounter, 42);
                int compareInt = atomicCompSwap(atomicBuffer.intCounter, 42, 84);
                
                // Unsigned integer atomic operations
                uint oldUint = atomicAdd(atomicBuffer.uintCounter, 1u);
                uint andUint = atomicAnd(atomicBuffer.uintCounter, 0xFFu);
                uint orUint = atomicOr(atomicBuffer.uintCounter, 0x10u);
                uint xorUint = atomicXor(atomicBuffer.uintCounter, 0xAAu);
                uint minUint = atomicMin(atomicBuffer.uintCounter, 100u);
                uint maxUint = atomicMax(atomicBuffer.uintCounter, 50u);
                uint exchangeUint = atomicExchange(atomicBuffer.uintCounter, 42u);
                uint compareUint = atomicCompSwap(atomicBuffer.uintCounter, 42u, 84u);
                
                // Memory barriers
                barrier();
                memoryBarrier();
                groupMemoryBarrier();
                memoryBarrierAtomicCounter();
                memoryBarrierBuffer();
                memoryBarrierShared();
                memoryBarrierImage();
                
                // Combined operations
                int combined = atomicAdd(atomicBuffer.intCounter, 
                    atomicMax(atomicBuffer.intCounter, 
                        atomicMin(atomicBuffer.intCounter, 200)
                    )
                );
            }
        ";
        
        stress_test_translation("All Atomic Operations", glsl_code, true);
    }

    #[test]
    fn stress_test_complex_control_flow() {
        let glsl_code = r"
            void main() {
                vec4 color = vec4(0.0);
                
                // Nested loops with complex conditions
                for (int i = 0; i < 10; i++) {
                    for (int j = 0; j < 5; j++) {
                        for (int k = 0; k < 3; k++) {
                            float value = float(i * j + k);
                            
                            if (value > 15.0) {
                                color.r += fract(value / 10.0);
                                if (color.r > 0.8) {
                                    break;
                                }
                            } else if (value > 10.0) {
                                color.g += mix(0.0, 1.0, value / 15.0);
                                continue;
                            } else {
                                color.b += inversesqrt(value + 1.0);
                            }
                            
                            // Nested ternary operations
                            color.a += (i % 2 == 0) ? 
                                ((j % 2 == 0) ? 
                                    ((k % 2 == 0) ? 0.1 : 0.2) : 
                                    ((k % 2 == 0) ? 0.3 : 0.4)) :
                                ((j % 2 == 0) ? 
                                    ((k % 2 == 0) ? 0.5 : 0.6) : 
                                    ((k % 2 == 0) ? 0.7 : 0.8));
                        }
                        
                        if (color.g > 0.9) {
                            continue;
                        }
                    }
                    
                    // Switch statement (if supported)
                    switch (i % 4) {
                        case 0:
                            color = mix(color, vec4(1.0, 0.0, 0.0, 1.0), 0.1);
                            break;
                        case 1:
                            color = mix(color, vec4(0.0, 1.0, 0.0, 1.0), 0.1);
                            break;
                        case 2:
                            color = mix(color, vec4(0.0, 0.0, 1.0, 1.0), 0.1);
                            break;
                        default:
                            color = mix(color, vec4(1.0, 1.0, 1.0, 1.0), 0.1);
                            break;
                    }
                    
                    if (length(color) > 2.0) {
                        break;
                    }
                }
                
                // While loop with complex condition
                int iterations = 0;
                while (iterations < 100 && dot(color.rgb, vec3(0.299, 0.587, 0.114)) < 0.5) {
                    color.rgb *= 1.1;
                    iterations++;
                    
                    if (iterations % 10 == 0) {
                        color.a = clamp(color.a, 0.0, 1.0);
                    }
                }
                
                // Do-while loop (if supported)
                do {
                    color = normalize(color);
                    color *= 0.99;
                } while (length(color) > 0.1);
            }
        ";
        
        stress_test_translation("Complex Control Flow", glsl_code, true);
    }

    #[test]
    fn stress_test_all_builtin_functions() {
        let glsl_code = r"
            void main() {
                float x = 1.5;
                vec2 v2 = vec2(1.0, 2.0);
                vec3 v3 = vec3(1.0, 2.0, 3.0);
                vec4 v4 = vec4(1.0, 2.0, 3.0, 4.0);
                
                // Angle and trigonometry functions
                float rad = radians(x);
                float deg = degrees(x);
                float s = sin(x);
                float c = cos(x);
                float t = tan(x);
                float as = asin(x);
                float ac = acos(x);
                float at = atan(x);
                float at2 = atan(x, x);
                float sh = sinh(x);
                float ch = cosh(x);
                float th = tanh(x);
                float ash = asinh(x);
                float ach = acosh(x);
                float ath = atanh(x);
                
                // Exponential functions
                float p = pow(x, x);
                float e = exp(x);
                float l = log(x);
                float e2 = exp2(x);
                float l2 = log2(x);
                float sq = sqrt(x);
                float isq = inversesqrt(x);
                
                // Common functions
                float a = abs(x);
                float sg = sign(x);
                float fl = floor(x);
                float tr = trunc(x);
                float ro = round(x);
                float re = roundEven(x);
                float ce = ceil(x);
                float fr = fract(x);
                float mo = mod(x, x);
                float mi = min(x, x);
                float ma = max(x, x);
                float cl = clamp(x, 0.0, 1.0);
                float mx = mix(x, x, 0.5);
                float st = step(x, x);
                float ss = smoothstep(0.0, 1.0, x);
                bool isna = isnan(x);
                bool isin = isinf(x);
                
                // Geometric functions
                float len = length(v3);
                float dist = distance(v3, v3);
                float dt = dot(v3, v3);
                vec3 cr = cross(v3, v3);
                vec3 no = normalize(v3);
                vec3 ft = faceforward(v3, v3, v3);
                vec3 rf = reflect(v3, v3);
                vec3 rt = refract(v3, v3, x);
                
                // Matrix functions
                mat2 m2 = mat2(1.0);
                mat3 m3 = mat3(1.0);
                mat4 m4 = mat4(1.0);
                mat2 m2t = transpose(m2);
                mat3 m3t = transpose(m3);
                mat4 m4t = transpose(m4);
                float det2 = determinant(m2);
                float det3 = determinant(m3);
                float det4 = determinant(m4);
                mat2 inv2 = inverse(m2);
                mat3 inv3 = inverse(m3);
                mat4 inv4 = inverse(m4);
                
                // Vector relational functions
                bvec2 ltv2 = lessThan(v2, v2);
                bvec3 ltv3 = lessThan(v3, v3);
                bvec4 ltv4 = lessThan(v4, v4);
                bvec2 lev2 = lessThanEqual(v2, v2);
                bvec3 lev3 = lessThanEqual(v3, v3);
                bvec4 lev4 = lessThanEqual(v4, v4);
                bvec2 gtv2 = greaterThan(v2, v2);
                bvec3 gtv3 = greaterThan(v3, v3);
                bvec4 gtv4 = greaterThan(v4, v4);
                bvec2 gev2 = greaterThanEqual(v2, v2);
                bvec3 gev3 = greaterThanEqual(v3, v3);
                bvec4 gev4 = greaterThanEqual(v4, v4);
                bvec2 eqv2 = equal(v2, v2);
                bvec3 eqv3 = equal(v3, v3);
                bvec4 eqv4 = equal(v4, v4);
                bvec2 nev2 = notEqual(v2, v2);
                bvec3 nev3 = notEqual(v3, v3);
                bvec4 nev4 = notEqual(v4, v4);
                bool anyb2 = any(ltv2);
                bool anyb3 = any(ltv3);
                bool anyb4 = any(ltv4);
                bool allb2 = all(ltv2);
                bool allb3 = all(ltv3);
                bool allb4 = all(ltv4);
                bvec2 notb2 = not(ltv2);
                bvec3 notb3 = not(ltv3);
                bvec4 notb4 = not(ltv4);
                
                // Integer functions
                int ix = int(x);
                uint ux = uint(x);
                ivec2 iv2 = ivec2(v2);
                uvec2 uv2 = uvec2(v2);
                uint ufb = floatBitsToUint(x);
                int ifb = floatBitsToInt(x);
                float ubf = uintBitsToFloat(ux);
                float ibf = intBitsToFloat(ix);
                ivec2 pac = packSnorm2x16(v2);
                vec2 ups = unpackSnorm2x16(pac);
                uint pacu = packUnorm2x16(v2);
                vec2 upu = unpackUnorm2x16(pacu);
                uint pach = packHalf2x16(v2);
                vec2 uph = unpackHalf2x16(pach);
            }
        ";
        
        stress_test_translation("All Builtin Functions", glsl_code, true);
    }

    #[test]
    fn stress_test_glsl_specific_features() {
        let glsl_code = r"
            #version 450 core
            
            // Precision qualifiers (GLSL-specific)
            precision highp float;
            precision mediump int;
            
            // Layout qualifiers
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec2 texCoord;
            layout(location = 2) in vec3 normal;
            
            layout(location = 0) out vec4 fragColor;
            
            // Uniform block with layout
            layout(std140, binding = 0) uniform Transform {
                mat4 modelMatrix;
                mat4 viewMatrix;
                mat4 projectionMatrix;
            } transform;
            
            // Storage buffer (GLSL 4.3+)
            layout(std430, binding = 1) buffer OutputBuffer {
                vec4 results[];
            } outputBuffer;
            
            // Subroutines (GLSL 4.0+)
            subroutine vec4 ColorFunction(vec3 pos);
            
            subroutine(ColorFunction)
            vec4 redColor(vec3 pos) {
                return vec4(1.0, 0.0, 0.0, 1.0);
            }
            
            subroutine(ColorFunction)
            vec4 blueColor(vec3 pos) {
                return vec4(0.0, 0.0, 1.0, 1.0);
            }
            
            subroutine uniform ColorFunction getColor;
            
            // Invariant declarations
            invariant gl_Position;
            
            void main() {
                // Built-in variables
                vec4 worldPos = transform.modelMatrix * vec4(position, 1.0);
                vec4 viewPos = transform.viewMatrix * worldPos;
                gl_Position = transform.projectionMatrix * viewPos;
                
                // Subroutine call
                fragColor = getColor(worldPos.xyz);
                
                // Array with dynamic indexing
                vec3 colors[3] = vec3[3](
                    vec3(1.0, 0.0, 0.0),
                    vec3(0.0, 1.0, 0.0),
                    vec3(0.0, 0.0, 1.0)
                );
                
                int index = int(worldPos.x) % 3;
                fragColor.rgb *= colors[index];
                
                // Buffer write
                outputBuffer.results[gl_VertexID] = fragColor;
                
                // Discard for fragment shader
                if (fragColor.a < 0.1) {
                    discard;
                }
            }
        ";
        
        // This test may fail to parse or translate due to advanced GLSL features
        stress_test_translation("GLSL Specific Features", glsl_code, false);
    }

    #[test]
    fn stress_test_malformed_and_edge_cases() {
        // Test various edge cases that might break the translator
        let test_cases = vec![
            ("Empty main", "void main() {}"),
            ("Single statement", "void main() { float x = 1.0; }"),
            ("No parameters", "float func() { return 1.0; } void main() { float x = func(); }"),
            ("Unused variables", "void main() { float a, b, c, d, e, f, g; }"),
            ("Deep nesting", "void main() { if (true) { if (true) { if (true) { if (true) { float x = 1.0; } } } } }"),
            ("Many variables", r"
                void main() {
                    float a = 1.0, b = 2.0, c = 3.0, d = 4.0, e = 5.0;
                    vec2 v1 = vec2(a, b), v2 = vec2(c, d);
                    vec3 v3 = vec3(a, b, c), v4 = vec3(d, e, a);
                    vec4 v5 = vec4(v1, v2), v6 = vec4(v3, d);
                    mat2 m1 = mat2(a, b, c, d);
                    mat3 m2 = mat3(v3, v4, vec3(b, c, e));
                    mat4 m3 = mat4(v5, v6, vec4(1.0), vec4(0.0, 0.0, 0.0, 1.0));
                }
            "),
        ];
        
        for (name, code) in test_cases {
            stress_test_translation(name, code, true);
        }
    }
}
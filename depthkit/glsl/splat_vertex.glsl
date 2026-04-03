// 3DGS Gaussian Splat renderer — vertex shader
// Instanced on a Grid SOP (1x1 quad, 2 triangles).
// Each instance is one Gaussian; data fetched from texture uniforms.
//
// NOTE: v1 uses isotropic billboard sizing (max scale * 3 sigma).
// Rotation quaternion is loaded but not yet applied to billboard orientation.
// TODO: Project 3D covariance (scale + rotation) to screen-space ellipse
// for proper anisotropic Gaussian rendering.

uniform sampler2D uPositionTex;   // R=x, G=y, B=z, A=1
uniform sampler2D uColorTex;      // R, G, B, A=1
uniform sampler2D uScaleOpacity;  // R=sx, G=sy, B=sz, A=opacity
uniform sampler2D uRotationTex;   // R=w, G=x, B=y, A=z (quaternion)

uniform int uTexWidth;
uniform int uNumGaussians;
uniform float uSplatScale;        // Global scale multiplier (default 1.0)

out Vertex {
    vec4 color;
    vec2 quadPos;    // [-1, 1] position on the quad
    float opacity;
} oVert;

// Fetch texel by linear index
vec4 fetchByIndex(sampler2D tex, int idx) {
    int row = idx / uTexWidth;
    int col = idx - row * uTexWidth;
    return texelFetch(tex, ivec2(col, row), 0);
}

void main() {
    int instanceID = TDInstanceID();

    // Skip padding instances beyond actual Gaussian count
    if (instanceID >= uNumGaussians) {
        gl_Position = vec4(0.0, 0.0, -9999.0, 1.0);
        oVert.opacity = 0.0;
        return;
    }

    // Fetch Gaussian attributes from data textures
    vec4 posData   = fetchByIndex(uPositionTex, instanceID);
    vec4 colorData = fetchByIndex(uColorTex, instanceID);
    vec4 scaleOp   = fetchByIndex(uScaleOpacity, instanceID);

    vec3 gaussPos = posData.xyz;
    vec3 scale    = scaleOp.xyz * uSplatScale;
    float opacity = scaleOp.w;

    // Quad vertex position in [-1, 1] (from Grid SOP)
    vec3 localPos = P;
    oVert.quadPos = localPos.xy;

    // View-space center of Gaussian
    vec4 viewCenter = uTDMat.cam * vec4(gaussPos, 1.0);

    // Billboard radius: use max scale component * 3 sigma
    float maxScale = max(max(scale.x, scale.y), scale.z);
    float billboardRadius = maxScale * 3.0;

    // Camera-facing quad: offset in view space
    vec3 offset = vec3(localPos.xy * billboardRadius, 0.0);
    vec4 viewPos = viewCenter + vec4(offset, 0.0);

    gl_Position = uTDMat.proj * viewPos;

    oVert.color = vec4(colorData.rgb, 1.0);
    oVert.opacity = opacity;
}

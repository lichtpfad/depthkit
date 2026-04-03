// 3DGS Gaussian Splat renderer — pixel (fragment) shader
// Renders a 2D Gaussian falloff on each billboard quad.

in Vertex {
    vec4 color;
    vec2 quadPos;
    float opacity;
} iVert;

layout(location = 0) out vec4 fragColor;

void main() {
    // Distance squared from quad center (quadPos is [-1, 1])
    float dist2 = dot(iVert.quadPos, iVert.quadPos);

    // Gaussian falloff: 3-sigma at quad edge
    // sigma = 1/3, so exponent = -0.5 * dist2 / (1/9) = -4.5 * dist2
    float alpha = exp(-4.5 * dist2);

    // Discard nearly-transparent fragments for performance
    if (alpha < 0.004) discard;

    alpha *= iVert.opacity;

    fragColor = vec4(iVert.color.rgb * alpha, alpha);
}

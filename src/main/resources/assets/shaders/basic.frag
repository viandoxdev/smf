#version 120

varying vec2 tc;

void main() {
    gl_FragColor = vec4(tc, 1.0, 1.0);
}
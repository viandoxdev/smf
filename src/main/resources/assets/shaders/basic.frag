#version 120

varying vec2 tc;

uniform sampler2D tex;

void main() {
    gl_FragColor = vec4(1.0, 1.0, 1.0, texture2D(tex, tc));
}
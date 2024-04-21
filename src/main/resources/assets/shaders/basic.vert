#version 120

attribute vec2 pos;
attribute vec2 tex_coords;

varying vec2 tc;

void main() {
    tc = tex_coords;
    gl_Position = vec4(pos, -0.2, 1.0);
}


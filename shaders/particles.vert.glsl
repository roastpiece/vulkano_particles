#version 450

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 velocity;

layout(binding = 0) uniform UniformBufferObject {
    float aspect;
} ubo;

void main() {
    gl_Position = vec4(position.x * ubo.aspect, position.y, 0.0, 1.0);
}
#version 450

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

struct Particle {
    vec2 position;
    vec2 velocity;
};

layout(set = 0, binding = 0) buffer Data {
    Particle data[];
} vertices;

layout(set = 1, binding = 0) uniform UniformBufferObject {
    vec2 target;
    float delta_time;
    float target_mass;
} ubo;

void main() {
    float particle_mass = 0.01;

    uint idx = gl_GlobalInvocationID.x;

    float gravity = (ubo.target_mass * particle_mass) / distance(ubo.target, vertices.data[idx].position);

    vec2 delta = normalize(ubo.target - vertices.data[idx].position) * gravity;

    vertices.data[idx].velocity += delta;
    vertices.data[idx].velocity *= 0.999 * (1-ubo.delta_time);

    vertices.data[idx].position += vertices.data[idx].velocity * ubo.delta_time;
}

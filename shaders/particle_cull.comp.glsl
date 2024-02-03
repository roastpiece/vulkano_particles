#version 450

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

struct Particle {
    vec2 position;
    vec2 velocity;
};

struct DrawCommand {
    uint vertex_count;
    uint instance_count;
    uint first_vertex;
    uint first_instance;
};

layout(set = 0, binding = 0) readonly buffer InputBuffer {
    Particle data[];
} verticesInput;

layout(set = 0, binding = 1) buffer OutputBuffer {
    Particle data[];
} verticesOutput;

layout(set = 0, binding = 2) buffer DrawBuffer {
    DrawCommand drawCommand;
};

layout(set = 0, binding = 3) readonly buffer VertexCount {
    uint vertexCount;
};

layout(set = 0, binding = 4) buffer DrawVertexCount {
    uint drawVertexCount;
};

layout(push_constant) uniform PushConstants {
    float aspect;
};

bool isOffScreen(Particle particle);
void main() {
    uint idx = gl_GlobalInvocationID.x;

    if (idx == 0) {
        atomicExchange(drawVertexCount, 0);
    }

    if (!isOffScreen(verticesInput.data[idx])) {
        uint drawVertexIdx = atomicAdd(drawVertexCount, 1);
        verticesOutput.data[drawVertexIdx] = verticesInput.data[idx];

        drawCommand.instance_count = 1;
        drawCommand.first_instance = 0;
        drawCommand.vertex_count = drawVertexCount;
        drawCommand.first_vertex = 0;
    }
}

bool isOffScreen(Particle particle) {
    if (particle.position.x*aspect < -1) {
        return true;
    } else if (particle.position.x*aspect > 1) {
        return true;
    } else if (particle.position.y < -1) {
        return true;
    } else if (particle.position.y > 1) {
        return true;
    } else {
        return false;
    }
}
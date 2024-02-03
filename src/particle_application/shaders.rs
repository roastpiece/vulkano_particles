pub mod vs_graphics {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/particles.vert.glsl"
    }
}

pub mod fs_graphics {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/graphics.frag.glsl"
    }
}

pub mod cs_particle_physics {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/particle_physics.comp.glsl"
    }
}

pub mod cs_cull {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/particle_cull.comp.glsl"
    }
}


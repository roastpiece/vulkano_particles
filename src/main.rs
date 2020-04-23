mod particle_application;

use particle_application::ParticleApplication;

fn main() {
    let app = ParticleApplication::new();
    app.run();
}

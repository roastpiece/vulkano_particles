use std::mem::size_of;
use std::ops::Mul;
use rand::Rng;
use std::sync::Arc;
use vulkano::buffer::{BufferAccess, BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, DrawIndirectCommand, DynamicState, PrimaryCommandBuffer, SubpassContents};
use vulkano::descriptor::descriptor_set::{FixedSizeDescriptorSetsPool, PersistentDescriptorSet};
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::device::{Device, DeviceExtensions, Features, Queue};
use vulkano::image::{ImageAccess, SwapchainImage};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{ComputePipeline, GraphicsPipeline};
use vulkano::swapchain::{
    self, AcquireError, ColorSpace, FullscreenExclusive, PresentMode, Surface, SurfaceTransform,
    Swapchain, SwapchainCreationError,
};
use vulkano::{sync};
use vulkano::sync::{FlushError, GpuFuture};
use vulkano_win::VkSurfaceBuild;
use winit::event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget};
use winit::window::{Fullscreen, Window, WindowBuilder};
use rayon::prelude::*;
use vulkano::format::FormatTy::Float;
use vulkano::image::view::ImageView;
use vulkano::render_pass::{Framebuffer, FramebufferAbstract, RenderPass, Subpass};
use winit::dpi::Pixel;
use log::debug;

pub struct ParticleApplication;

impl ParticleApplication {
    pub fn new() -> ParticleApplication {
        ParticleApplication
    }

    pub fn run(&self) {
        let (device, queue, instance) = init_vulkan();
        vulkano_particles(device.clone(), queue.clone(), instance.clone());
    }
}

fn init_vulkan() -> (Arc<Device>, Arc<Queue>, Arc<Instance>) {
    println!("INIT VULKAN");

    let instance = Instance::new(None, &vulkano_win::required_extensions(), None)
        .expect("failed to create instance");

    let physical_dev = {
        PhysicalDevice::enumerate(&instance)
            .next()
            .expect("no device available")
    };

    println!("Physical Device: {}", physical_dev.name());

    let queue_family = physical_dev
        .queue_families()
        .find(|&q| q.supports_graphics() && q.supports_compute())
        .expect("couldn't find a queue with graphical and compute capabilities");

    let (device, mut queues) = {
        Device::new(
            physical_dev,
            &Features {
                fill_mode_non_solid: true,
                ..Features::none()
            },
            &DeviceExtensions {
                khr_storage_buffer_storage_class: true,
                khr_swapchain: true,
                ..DeviceExtensions::none()
            },
            [(queue_family, 0.5)].iter().cloned(),
        )
            .expect("failed to create device")
    };

    let queue = queues.next().unwrap();

    (device.clone(), queue.clone(), instance.clone())
}

pub fn vulkano_particles(device: Arc<Device>, queue: Arc<Queue>, instance: Arc<Instance>) {
    let events_loop = EventLoop::new();

    let mut surface = build_window(None, &events_loop, instance.clone());

    let caps = surface
        .capabilities(device.physical_device())
        .expect("failed to get surface capabilities");

    let dimensions = caps.current_extent.unwrap_or([1280, 1024]);
    let alpha = caps.supported_composite_alpha.iter().next().unwrap();
    let format = caps.supported_formats[0].0;

    let (mut swapchain, images) = Swapchain::start(device.clone(),surface.clone())
        .format(format)
        .dimensions(dimensions)
        .num_images(caps.min_image_count)
        .usage(caps.supported_usage_flags)
        .composite_alpha(alpha)
        .build().unwrap();

    println!("Generating particles");
    const PARTICLE_COUNT: u32 = 1_000_000;
    let mut particles = vec![Vertex::new(0f32, 0f32); PARTICLE_COUNT as usize];
    particles.par_iter_mut().for_each(|v| v.position = [rand::thread_rng().gen_range(-1.0..1.0), rand::thread_rng().gen_range(-1.0..1.0)]);
    println!("Done. Enjoy");

    let cpu_vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage {
            transfer_source: true,
            ..BufferUsage::none()
        },
        false,
        particles.into_iter(),
    )
        .unwrap();

    let vertex_buffer = DeviceLocalBuffer::<[Vertex]>::array(
        device.clone(),
        cpu_vertex_buffer.len(),
        BufferUsage {
            transfer_destination: true,
            vertex_buffer: true,
            storage_buffer: true,
            ..BufferUsage::none()
        },
        vec![queue.family()],
    )
        .unwrap();

    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        )
            .unwrap(),
    );

    let draw_indirect_cmd_buffers = vec![unsafe {
        DeviceLocalBuffer::<[DrawIndirectCommand]>::raw(
            device.clone(),
            16,
            BufferUsage {
                indirect_buffer: true,
                storage_buffer: true,
                ..BufferUsage::indirect_buffer()
            },
            vec![queue.family()],
        )
    }.unwrap(),
                                         unsafe {
        DeviceLocalBuffer::<[DrawIndirectCommand]>::raw(
            device.clone(),
            16,
            BufferUsage {
                indirect_buffer: true,
                storage_buffer: true,
                ..BufferUsage::indirect_buffer()
            },
            vec![queue.family()],
        )
    }.unwrap()];

    let draw_output_buffers = vec![DeviceLocalBuffer::array(
        device.clone(),
        1920*1080, // should be fine up to 4k
        BufferUsage {
            vertex_buffer: true,
            storage_buffer: true,
            ..BufferUsage::none()
        },
        vec![queue.family()]
    ).unwrap(),DeviceLocalBuffer::<[Vertex]>::array(
        device.clone(),
        1920*1080, // should be fine up to 4k
        BufferUsage {
            vertex_buffer: true,
            storage_buffer: true,
            ..BufferUsage::none()
        },
        vec![queue.family()]
    ).unwrap()];
    let mut current_update_buffer = 0;

    let cpu_vertex_count_buffer = CpuAccessibleBuffer::from_data(
        device.clone(),
        BufferUsage {
            transfer_source: true,
            ..BufferUsage::none()
        },
        false,
        [PARTICLE_COUNT]
    ).unwrap();
    let vertex_count_buffer = unsafe {
        DeviceLocalBuffer::<[u32; 1]>::raw(
            device.clone(),
            4,
            BufferUsage {
                storage_buffer: true,
                transfer_destination: true,
                ..BufferUsage::none()
            },
            vec![queue.family()]
        )
    }.unwrap();
    let draw_vertex_count_buffer = unsafe {
        DeviceLocalBuffer::<Vec<u32>>::raw(
            device.clone(),
            4,
            BufferUsage {
                storage_buffer: true,
                transfer_destination: true,
                ..BufferUsage::none()
            },
            vec![queue.family()]
        )
    }.unwrap();

    {
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(device.clone(), queue.clone().family(), CommandBufferUsage::OneTimeSubmit).unwrap();
        command_buffer_builder.copy_buffer(cpu_vertex_buffer.clone(), vertex_buffer.clone()).unwrap()
            .copy_buffer(cpu_vertex_count_buffer.clone(), vertex_count_buffer.clone()).unwrap();

        let command_buffer = command_buffer_builder.build().unwrap();

        command_buffer.execute(queue.clone()).unwrap()
            .then_signal_fence_and_flush().unwrap()
            .wait(None).unwrap();
    }

    let mut dynamic_state = DynamicState::none();

    let mut framebuffers =
        window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

    let vert_shader =
        shaders::vs_graphics::Shader::load(device.clone()).expect("failed to create vert_shader");
    let frag_shader =
        shaders::fs_graphics::Shader::load(device.clone()).expect("failed to create frag_shader");

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .polygon_mode_point()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vert_shader.main_entry_point(), ())
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(frag_shader.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    let vertex_uniform_layout = pipeline.layout().descriptor_set_layout(0).unwrap();
    let mut vertex_uniform_pool = FixedSizeDescriptorSetsPool::new(vertex_uniform_layout.clone());

    let particle_shader = shaders::cs_particle_physics::Shader::load(device.clone())
        .expect("failed to load particle_shader");

    let particle_compute_pipeline = Arc::new(
        ComputePipeline::new(device.clone(), &particle_shader.main_entry_point(), &(), None)
            .expect("failed to create particle_compute_pipeline"),
    );

    let vertex_layout = particle_compute_pipeline
        .layout()
        .descriptor_set_layout(0)
        .unwrap();

    let uniform_layout = particle_compute_pipeline
        .layout()
        .descriptor_set_layout(1)
        .unwrap();

    let particle_set = Arc::new(
        PersistentDescriptorSet::start(vertex_layout.clone())
            .add_buffer(vertex_buffer.clone())
            .unwrap()
            .build()
            .unwrap(),
    );

    let cull_shader = shaders::cs_cull::Shader::load(device.clone())
        .expect("failed to load cull_shader");

    let cull_compute_pipeline = Arc::new(
        ComputePipeline::new(device.clone(), &cull_shader.main_entry_point(), &(), None)
            .expect("failed to create cull_compute_pipeline"),
    );

    let cull_layout = cull_compute_pipeline
        .layout()
        .descriptor_set_layout(0)
        .unwrap();

    let cull_descriptor_sets = vec![Arc::new(
        PersistentDescriptorSet::start(cull_layout.clone())
            .add_buffer(vertex_buffer.clone()).unwrap()
            .add_buffer(draw_output_buffers[0].clone()).unwrap()
            .add_buffer(draw_indirect_cmd_buffers[0].clone()).unwrap()
            .add_buffer(vertex_count_buffer.clone()).unwrap()
            .add_buffer(draw_vertex_count_buffer.clone()).unwrap()
            .build().unwrap()
    ),Arc::new(
        PersistentDescriptorSet::start(cull_layout.clone())
            .add_buffer(vertex_buffer.clone()).unwrap()
            .add_buffer(draw_output_buffers[1].clone()).unwrap()
            .add_buffer(draw_indirect_cmd_buffers[1].clone()).unwrap()
            .add_buffer(vertex_count_buffer.clone()).unwrap()
            .add_buffer(draw_vertex_count_buffer.clone()).unwrap()
            .build().unwrap()
    )];

    let mut uniform_pool = FixedSizeDescriptorSetsPool::new(uniform_layout.clone());

    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);
    let mut recreate_swapchain = false;

    let mut mouse_position: [f32; 2] = [0.0, 0.0];

    let mut last_time = std::time::Instant::now();
    let mut delta_time: f32 = 0.0;
    let mut target_mass: f32 = 1.0;

    let mut aspect: f32 = 1.0;

    events_loop.run(move |event, events_loop, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                winit::event::WindowEvent::CursorMoved { position, .. } => {
                    let dimensions = surface.window().inner_size();
                    mouse_position = [
                        (((position.x / dimensions.width as f64) - 0.5) * 2.0 / aspect as f64)
                            as f32,
                        (((position.y / dimensions.height as f64) - 0.5) * 2.0) as f32,
                    ];
                }
                winit::event::WindowEvent::MouseWheel { delta, .. } => match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, dy) => {
                        target_mass += dy * 0.05;
                        println!("Mass: {:.2}", target_mass);
                    }
                    _ => (),
                },
                winit::event::WindowEvent::KeyboardInput {
                    input:
                        winit::event::KeyboardInput {
                            state,
                            virtual_keycode: Some(kc),
                            ..
                        },
                    ..
                } => match (kc, state) {
                    (winit::event::VirtualKeyCode::Escape, winit::event::ElementState::Pressed) => {
                        *control_flow = ControlFlow::Exit;
                    }
                    (winit::event::VirtualKeyCode::F, winit::event::ElementState::Pressed) => {
                        match surface.window().fullscreen() {
                            Some(_) => surface.window().set_fullscreen(None),
                            None => {
                                let fullscreen = Fullscreen::Exclusive(
                                    surface
                                        .window()
                                        .primary_monitor()
                                        .unwrap()
                                        .video_modes()
                                        .next()
                                        .unwrap(),
                                );
                                surface.window().set_fullscreen(Some(fullscreen));
                            }
                        };
                        recreate_swapchain = true;
                    }
                    _ => (),
                },
                _ => (),
            },
            winit::event::Event::MainEventsCleared => {
                let time = std::time::Instant::now();
                let delta_time_instant = time - last_time;

                delta_time = delta_time_instant.as_secs_f32();
                last_time = time;

                let uniform_data = ParticleUBO {
                    target: mouse_position.clone(),
                    delta_time,
                    target_mass,
                };

                let particle_uniform_buffer = CpuAccessibleBuffer::from_data(
                    device.clone(),
                    BufferUsage::all(),
                    false,
                    uniform_data,
                )
                    .expect("failed to create particle_uniform_buffer");
                let uniform_set = uniform_pool
                    .next()
                    .add_buffer(particle_uniform_buffer.clone()).unwrap()
                    .build().unwrap();


                let mut particle_cmd_buffer_builder =
                    AutoCommandBufferBuilder::primary(device.clone(), queue.clone().family(), CommandBufferUsage::OneTimeSubmit).unwrap();
                particle_cmd_buffer_builder
                        .dispatch(
                            [PARTICLE_COUNT / 1024, 1, 1],
                            particle_compute_pipeline.clone(),
                            (particle_set.clone(), uniform_set),
                            (),
                            vec![]
                        ).unwrap();
                let particle_cmd_buffer = particle_cmd_buffer_builder.build().unwrap();
                let particle_cmd_future = particle_cmd_buffer.execute(queue.clone()).unwrap()
                    .then_signal_fence_and_flush().unwrap();

                let mut cull_cmd_buffer_builder =
                    AutoCommandBufferBuilder::primary(device.clone(), queue.clone().family(), CommandBufferUsage::OneTimeSubmit).unwrap();
                cull_cmd_buffer_builder
                        .dispatch(
                            [PARTICLE_COUNT / 1024, 1, 1],
                            cull_compute_pipeline.clone(),
                            cull_descriptor_sets[current_update_buffer].clone(),
                            vec![aspect],
                            vec![]
                        ).unwrap();
                let cull_cmd_buffer = cull_cmd_buffer_builder.build().unwrap();
                let cull_cmd_future = cull_cmd_buffer.execute_after(particle_cmd_future, queue.clone()).unwrap();
                cull_cmd_future
                    .then_signal_fence_and_flush().unwrap()
                    .wait(None).unwrap();

                surface
                    .window()
                    .set_title(format!("FPS: {:.2}", 1.0 / delta_time).as_str());

                surface.window().request_redraw();
            }
            winit::event::Event::RedrawRequested(_) => {
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    match recreate_swap_chain(
                        &surface,
                        render_pass.clone(),
                        swapchain.clone(),
                        &mut dynamic_state,
                        &mut aspect,
                    ) {
                        Ok((new_swapchain, new_framebuffrers)) => {
                            swapchain = new_swapchain;
                            framebuffers = new_framebuffrers;
                        }
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                    };
                    recreate_swapchain = false;
                }

                let (image_num, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let vertex_uniform_buffer = CpuAccessibleBuffer::from_data(
                    device.clone(),
                    BufferUsage::uniform_buffer(),
                    false,
                    VertexUBO { aspect },
                )
                    .unwrap();

                let vertex_uniform_set = vertex_uniform_pool
                    .next()
                    .add_buffer(vertex_uniform_buffer.clone()).unwrap()
                    .build().unwrap();

                let mut draw_command_buffer_builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.family(),
                    CommandBufferUsage::OneTimeSubmit
                ).unwrap();
                draw_command_buffer_builder
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        vec![[0.0, 0.0, 0.0, 1.0].into()],
                    ).unwrap()
                    .draw_indirect(
                        pipeline.clone(),
                        &dynamic_state,
                        draw_output_buffers[current_update_buffer].clone(),
                        draw_indirect_cmd_buffers[current_update_buffer].clone(),
                        vertex_uniform_set,
                        (),
                        vec![]
                    ).unwrap()
                    .end_render_pass().unwrap();
                let draw_command_buffer = draw_command_buffer_builder.build().unwrap();

                let future = previous_frame_end
                    .take().unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), draw_command_buffer).unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                current_update_buffer = (current_update_buffer + 1) % 2;

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(Box::new(future) as Box<_>);
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                    }
                    Err(e) => {
                        println!("Failed to flush future {:?}", e);
                        previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                    }
                }
            }
            _ => (),
        }
    });
}

fn build_window(
    fullscreen: Option<Fullscreen>,
    events_loop: &EventLoopWindowTarget<()>,
    instance: Arc<Instance>,
) -> Arc<Surface<Window>> {
    match fullscreen {
        Some(fullscreen) => WindowBuilder::new()
            .with_fullscreen(Some(fullscreen))
            .build_vk_surface(events_loop, instance.clone())
            .unwrap(),
        None => WindowBuilder::new()
            .build_vk_surface(events_loop, instance.clone())
            .unwrap(),
    }
}

fn recreate_swap_chain(
    surface: &Surface<Window>,
    render_pass: Arc<RenderPass>,
    swapchain: Arc<Swapchain<Window>>,
    dynamic_state: &mut DynamicState,
    aspect: &mut f32,
) -> Result<
    (
        Arc<Swapchain<Window>>,
        Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
    ),
    SwapchainCreationError,
> {
    let dimensions: [u32; 2] = surface.window().inner_size().into();

    *aspect = dimensions[1] as f32 / dimensions[0] as f32;

    println!("New aspect ratio {}", aspect);

    let (new_swapchain, new_images) = match swapchain.recreate().dimensions(dimensions).build() {
        Ok(r) => r,
        Err(e) => return Err(e),
    };

    Ok((
        new_swapchain,
        window_size_dependent_setup(&new_images, render_pass.clone(), dynamic_state),
    ))
}

fn window_size_dependent_setup(
    images: &Vec<Arc<SwapchainImage<Window>>>,
    render_pass: Arc<RenderPass>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: dimensions.width_height().map(f64::from).map(f32::from_f64),
        depth_range: 0.0..1.0,
    };

    dynamic_state.viewports = Some(vec![viewport]);

    images
        .iter()
        .map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(ImageView::new(image.clone()).unwrap()).unwrap()
                    .build().unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}

#[derive(Default, Copy, Clone)]
pub struct Vertex {
    position: [f32; 2],
    velocity: [f32; 2],
}

impl Vertex {
    fn new(x: f32, y: f32) -> Vertex {
        Vertex {
            position: [x, y],
            velocity: [0.0, 0.0],
        }
    }
}
vulkano::impl_vertex!(Vertex, position, velocity);

#[derive(Copy, Clone)]
struct ParticleUBO {
    target: [f32; 2],
    delta_time: f32,
    target_mass: f32,
}

#[derive(Copy, Clone)]
struct VertexUBO {
    aspect: f32,
}

mod shaders;

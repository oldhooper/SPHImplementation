#define _USE_MATH_DEFINES

//#include "C:\Users\admin\dev\SPHImplementation\tmp\libs\nanogui\ext\glad\include\glad\glad.h"
#include <nanogui/nanogui.h>
//#include <oneapi/tbb/parallel_for.h>
//#include <oneapi/tbb/blocked_range.h>
//#include <oneapi/tbb/concurrent_hash_map.h>
#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <chrono>

using namespace nanogui;

// Constants
constexpr double H = 16.0;
constexpr double HSQ = H * H;
constexpr double DT = 0.0007;
constexpr double REST_DENS = 300.0;
constexpr double GAS_CONST = 2000.0;
constexpr double VISC = 200.0;
constexpr double MASS = 2.5;
constexpr double GX = 0.0, GY = -10.0;
constexpr double EPS = H;
constexpr double BOUND_DAMPING = -0.5;
constexpr double CELL_SIZE = 2 * H;
constexpr int BOX_WIDTH = 800;
constexpr int BOX_HEIGHT = 600;

// Precomputed powers of H for constexpr calculations
constexpr double H_POW_5 = H * H * H * H * H;              // H^5
constexpr double H_POW_8 = H_POW_5 * H * H * H;            // H^8

// 2D Kernels - теперь constexpr так как используем предвычисленные степени
const double POLY6 = 4.0 / (M_PI * H_POW_8);
const double SPIKY_GRAD = -10.0 / (M_PI * H_POW_5);
const double VISC_LAP = 40.0 / (M_PI * H_POW_5);

struct Vector2 {
    double x = 0.0, y = 0.0;

    Vector2 operator+(const Vector2& o) const { return { x + o.x, y + o.y }; }
    Vector2 operator-(const Vector2& o) const { return { x - o.x, y - o.y }; }
    Vector2 operator*(double s) const { return { x * s, y * s }; }
    Vector2 operator/(double s) const { return { x / s, y / s }; }
    Vector2& operator+=(const Vector2& o) { x += o.x; y += o.y; return *this; }
    Vector2& operator-=(const Vector2& o) { x -= o.x; y -= o.y; return *this; }
    Vector2& operator*=(double s) { x *= s; y *= s; return *this; }
    double magnitude() const { return std::sqrt(x * x + y * y); }
    double magnitudeSquared() const { return x * x + y * y; }
    Vector2 normalized() const {
        double m = magnitude();
        return m > 0.0 ? Vector2{x / m, y / m} : Vector2{ 0.0, 0.0 };
    }
};

struct Particle {
    Vector2 pos, vel, force;
    double rho = REST_DENS;
    double p = 0.0;
};

struct CellKeyHashCompare {
    static size_t hash(const std::array<int, 2>& k) {
        return static_cast<size_t>(k[0]) ^ (static_cast<size_t>(k[1]) << 1);
    }
    static bool equal(const std::array<int, 2>& a, const std::array<int, 2>& b) {
        return a == b;
    }
};

class SPHScreen : public Screen {
public:
    std::vector<Particle> particles;
    bool running = true;
    bool needs_redraw = true;
    Label* fps_label = nullptr;

    SPHScreen() : Screen(Vector2i(BOX_WIDTH, BOX_HEIGHT ), "2D SPH with NanoGUI") {
        // Initialize dam break
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> jitter(-0.5, 0.5);

        // Create a dam break setup - particles in the left part of the container
        int cols = 30;
        int rows = 20;
        particles.resize(cols * rows);
        int idx = 0;

        // Fill particles in a rectangular block on the left
        double start_x = 100.0;
        double start_y = 200.0;
        double spacing = H * 0.6;

        for (int i = 0; i < cols; ++i) {
            for (int j = 0; j < rows; ++j) {
                particles[idx].pos = Vector2{
                    start_x + i * spacing + jitter(gen),
                    start_y + j * spacing + jitter(gen)
                };
                particles[idx].rho = REST_DENS;
                idx++;
            }
        }

        // Buttons for control
        Window* window = new Window(this, "Control Panel");
        window->setPosition(Vector2i(10, 10));
        window->setLayout(new GroupLayout());

        Button* btn_reset = new Button(window, "Reset Simulation");
        btn_reset->setCallback([this]() { resetSimulation(); });

        Button* btn_pause = new Button(window, "Pause/Resume");
        btn_pause->setCallback([this]() {
            running = !running;
            needs_redraw = true;
            });

        fps_label = new Label(window, "FPS: 0");

        performLayout();

        glfwSetTime(0.0);
    }

    void resetSimulation() {
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> jitter(-0.5, 0.5);

        int cols = 30;
        int rows = 20;
        double start_x = 100.0;
        double start_y = 200.0;
        double spacing = H * 0.6;

        for (size_t i = 0; i < particles.size(); ++i) {
            int col = static_cast<int>(i) / rows;
            int row = static_cast<int>(i) % rows;

            particles[i].pos = Vector2{
                start_x + col * spacing + jitter(gen),
                start_y + row * spacing + jitter(gen)
            };

            particles[i].vel = Vector2{ 0.0, 0.0 };
            particles[i].force = Vector2{ 0.0, 0.0 };
            particles[i].rho = REST_DENS;
            particles[i].p = 0.0;
        }
        needs_redraw = true;
    }

    virtual bool keyboardEvent(int key, int scancode, int action, int modifiers) override {
        if (Screen::keyboardEvent(key, scancode, action, modifiers)) return true;
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            setVisible(false);
            return true;
        }
        if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
            running = !running;
            needs_redraw = true;
            return true;
        }
        if (key == GLFW_KEY_R && action == GLFW_PRESS) {
            resetSimulation();
            return true;
        }
        return false;
    }

    virtual void draw(NVGcontext* ctx) override {
        static auto last_time = std::chrono::high_resolution_clock::now();
        static int frame_count = 0;

        // Update FPS counter every second
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time).count();

        frame_count++;
        if (elapsed > 1000) {
            float fps = frame_count * 1000.0f / elapsed;
            if (fps_label) {
                fps_label->setCaption("FPS: " + std::to_string(static_cast<int>(fps)));
            }
            frame_count = 0;
            last_time = current_time;
        }

        Screen::draw(ctx);
    }

    virtual void drawContents() override {
        static auto last_update = glfwGetTime();
        double current_time = glfwGetTime();
        double delta_time = current_time - last_update;

        // Update simulation at fixed time steps
        if (running && delta_time > 0.016) {
            updateSPH();
            last_update = current_time;
            needs_redraw = true;
        }

        if (needs_redraw) {
            drawParticles();
            needs_redraw = false;
        }
    }

private:
    void drawParticles() {
        // Rendering with NanoVG
        NVGcontext* vg = mNVGContext;
        nvgBeginFrame(vg, mSize.x(), mSize.y(), 1.0f);

        // Background
        nvgBeginPath(vg);
        nvgRect(vg, 0, 0, mSize.x(), mSize.y());
        nvgFillColor(vg, nvgRGBA(20, 20, 30, 255));
        nvgFill(vg);

        // Draw boundaries
        nvgBeginPath(vg);
        nvgRect(vg, 0, 0, mSize.x(), mSize.y());
        nvgStrokeColor(vg, nvgRGBA(255, 255, 255, 100));
        nvgStrokeWidth(vg, 2.0f);
        nvgStroke(vg);

        // Particles
        for (const auto& p : particles) {
            float radius = 3.5f;

            // Color based on pressure
            double pressure_ratio = std::min(1.0, std::abs(p.p) / (GAS_CONST * REST_DENS));
            int red = static_cast<int>(255 * pressure_ratio);
            int green = static_cast<int>(255 * (1.0 - pressure_ratio * 0.5));
            int blue = static_cast<int>(255 * (1.0 - pressure_ratio));

            // Draw particle
            nvgBeginPath(vg);
            nvgCircle(vg, p.pos.x, BOX_HEIGHT - p.pos.y, radius);

            // Create gradient
            NVGpaint paint = nvgRadialGradient(vg,
                p.pos.x,
                BOX_HEIGHT - p.pos.y,
                0,
                radius * 1.5f,
                nvgRGBA(red, green, blue, 220),
                nvgRGBA(red / 3, green / 3, blue / 3, 50));

            nvgFillPaint(vg, paint);
            nvgFill(vg);
        }

        // Draw info text
        nvgFontSize(vg, 14.0f);
        nvgFontFace(vg, "sans");
        nvgFillColor(vg, nvgRGBA(255, 255, 255, 200));
        nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_TOP);
        nvgText(vg, 10, 10, "SPH Fluid Simulation", nullptr);
        nvgText(vg, 10, 30, "Controls: SPACE=Pause, R=Reset, ESC=Exit", nullptr);

        nvgEndFrame(vg);
    }

    void updateSPH() {
#if 0
        // Build grid for neighbors
        oneapi::tbb::concurrent_hash_map<std::array<int, 2>, std::vector<int>, CellKeyHashCompare> grid;

        // Insert particles into grid
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, particles.size()),
            [&](const oneapi::tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    const auto& pos = particles[i].pos;
                    std::array<int, 2> key = {
                        static_cast<int>(std::floor(pos.x / CELL_SIZE)),
                        static_cast<int>(std::floor(pos.y / CELL_SIZE))
                    };
                    oneapi::tbb::concurrent_hash_map<std::array<int, 2>, std::vector<int>, CellKeyHashCompare>::accessor acc;
                    if (grid.insert(acc, key)) {
                        acc->second.reserve(32);
                    }
                    acc->second.push_back(static_cast<int>(i));
                }
            });

        // Compute density and pressure
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, particles.size()),
            [&](const oneapi::tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    auto& p_i = particles[i];
                    const auto key = std::array<int, 2>{
                        static_cast<int>(std::floor(p_i.pos.x / CELL_SIZE)),
                            static_cast<int>(std::floor(p_i.pos.y / CELL_SIZE))
                    };

                    double rho = 0.0;

                    // Check neighboring cells
                    for (int dx = -1; dx <= 1; ++dx) {
                        for (int dy = -1; dy <= 1; ++dy) {
                            std::array<int, 2> nkey = { key[0] + dx, key[1] + dy };
                            oneapi::tbb::concurrent_hash_map<std::array<int, 2>, std::vector<int>, CellKeyHashCompare>::const_accessor cacc;
                            if (grid.find(cacc, nkey)) {
                                for (int j : cacc->second) {
                                    const auto& p_j = particles[j];
                                    double dist_sq = (p_i.pos.x - p_j.pos.x) * (p_i.pos.x - p_j.pos.x) +
                                        (p_i.pos.y - p_j.pos.y) * (p_i.pos.y - p_j.pos.y);

                                    if (dist_sq < HSQ) {
                                        double h_diff = HSQ - dist_sq;
                                        rho += MASS * POLY6 * h_diff * h_diff * h_diff;
                                    }
                                }
                            }
                        }
                    }

                    p_i.rho = std::max(rho, REST_DENS * 0.1);
                    p_i.p = GAS_CONST * (p_i.rho - REST_DENS);
                }
            });

        // Compute forces
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, particles.size()),
            [&](const oneapi::tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    auto& p_i = particles[i];
                    const auto key = std::array<int, 2>{
                        static_cast<int>(std::floor(p_i.pos.x / CELL_SIZE)),
                            static_cast<int>(std::floor(p_i.pos.y / CELL_SIZE))
                    };

                    Vector2 f_press{ 0.0, 0.0 };
                    Vector2 f_visc{ 0.0, 0.0 };

                    for (int dx = -1; dx <= 1; ++dx) {
                        for (int dy = -1; dy <= 1; ++dy) {
                            std::array<int, 2> nkey = { key[0] + dx, key[1] + dy };
                            oneapi::tbb::concurrent_hash_map<std::array<int, 2>, std::vector<int>, CellKeyHashCompare>::const_accessor cacc;
                            if (grid.find(cacc, nkey)) {
                                for (int j : cacc->second) {
                                    if (i == static_cast<size_t>(j)) continue;

                                    const auto& p_j = particles[j];
                                    double dist_sq = (p_i.pos.x - p_j.pos.x) * (p_i.pos.x - p_j.pos.x) +
                                        (p_i.pos.y - p_j.pos.y) * (p_i.pos.y - p_j.pos.y);

                                    if (dist_sq < HSQ && dist_sq > 1e-12) {
                                        double r = std::sqrt(dist_sq);
                                        Vector2 dir = (p_i.pos - p_j.pos) / r;

                                        // Pressure force (corrected kernel)
                                        double pres_kernel = SPIKY_GRAD * (H - r) * (H - r);
                                        double shared_pressure = MASS * (p_i.p / (p_i.rho * p_i.rho) + p_j.p / (p_j.rho * p_j.rho));
                                        f_press += dir * (-shared_pressure * pres_kernel);

                                        // Viscosity force
                                        Vector2 v_diff = p_j.vel - p_i.vel;
                                        double visc_kernel = VISC_LAP * (H - r);
                                        f_visc += v_diff * (VISC * MASS * visc_kernel / p_j.rho);
                                    }
                                }
                            }
                        }
                    }

                    // Gravity force
                    Vector2 f_grav = { GX * p_i.rho, GY * p_i.rho };
                    p_i.force = f_press + f_visc + f_grav;
                }
            });

        // Integrate motion
        for (auto& p : particles) {
            // Acceleration from forces
            Vector2 accel = p.force / p.rho;

            // Semi-implicit Euler integration
            p.vel += accel * DT;
            p.pos += p.vel * DT;

            // Boundary collisions with damping
            const double damping = 0.3;

            // Left boundary
            if (p.pos.x < EPS) {
                p.vel.x = -p.vel.x * damping;
                p.pos.x = EPS + 0.001;
            }
            // Right boundary
            if (p.pos.x > BOX_WIDTH - EPS) {
                p.vel.x = -p.vel.x * damping;
                p.pos.x = BOX_WIDTH - EPS - 0.001;
            }
            // Bottom boundary
            if (p.pos.y < EPS) {
                p.vel.y = -p.vel.y * damping;
                p.pos.y = EPS + 0.001;
            }
            // Top boundary
            if (p.pos.y > BOX_HEIGHT - EPS) {
                p.vel.y = -p.vel.y * damping;
                p.pos.y = BOX_HEIGHT - EPS - 0.001;
            }
        }
#endif
    }
};

int main(int argc, char** argv) {
    try {
        nanogui::init();

        {
            ref<SPHScreen> screen = new SPHScreen();
            screen->drawAll();
            screen->setVisible(true);

            // Use NanoGUI's mainloop
            nanogui::mainloop();
        }

        nanogui::shutdown();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
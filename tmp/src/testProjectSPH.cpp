#define _USE_MATH_DEFINES
#define NOMINMAX          // prevent Windows near/far/min/max macros

#include <nanogui/nanogui.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/concurrent_hash_map.h>
#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <string>

using namespace nanogui;

constexpr int    BOX_W = 300;
constexpr int    BOX_H = 600;

constexpr int    DEF_COLS = 18;
constexpr int    DEF_ROWS = 25;
constexpr double DEF_SPACING = 2.0;
constexpr double DEF_H = 6.0;   // ≈ 2.67 × SPACING
constexpr double DEF_DT = 0.008;  // CFL: H/(2c) = 16/(2×283) ≈ 0.028 → use 0.008
constexpr int    DEF_MSTEPS = 4;
constexpr double REST_DENS = 1000.0;
constexpr double DEF_GAS = 8e4;    // softer fluid → fewer springy oscillations
constexpr double DEF_VISC = 300.0;
constexpr double DEF_GY = -9800.0;
constexpr double DEF_DAMP = 0.992;  // global velocity decay per physics step
constexpr double EPS = 2.0;    // wall inset
constexpr double WALL_DAMP = 0.1;    // fraction of normal velocity kept after bounce

// Vector str

struct V2 {
    double x = 0, y = 0;

    V2 operator+(const V2& o) const { return { x + o.x, y + o.y }; }
    V2 operator-(const V2& o) const { return { x - o.x, y - o.y }; }
    V2 operator*(double s)    const { return { x * s,   y * s }; }
    V2 operator/(double s)    const { return { x / s,   y / s }; }
    V2& operator+=(const V2& o) { x += o.x; y += o.y; return *this; }
    V2& operator*=(double s) { x *= s;   y *= s;   return *this; }

    double magSq() const { return x * x + y * y; }
    double mag()   const { return std::sqrt(magSq()); }
    V2 norm()      const { double m = mag(); return m > 1e-12 ? V2{x / m, y / m} : V2{}; }
};

// Particle str

struct Particle {
    V2     pos{}, vel{}, force{};
    double rho = REST_DENS;
    double p = 0.0;
};

// Hash str

struct CellHash {
    static size_t hash(const std::array<int, 2>& k) {
        return static_cast<size_t>(k[0]) ^ (static_cast<size_t>(k[1]) * 2654435761ULL);
    }
    static bool equal(const std::array<int, 2>& a, const std::array<int, 2>& b) {
        return a == b;
    }
};

using Grid = oneapi::tbb::concurrent_hash_map<std::array<int, 2>, std::vector<int>, CellHash>;



class SPHScreen : public Screen {
public:
    std::vector<Particle> particles;
    bool running = true;

    Label* lbl_fps = nullptr;
    Label* lbl_count = nullptr;

    double p_spacing = DEF_SPACING;
    double p_h = DEF_H;
    double p_dt = DEF_DT;
    double p_gas = DEF_GAS;
    double p_visc = DEF_VISC;
    double p_gy = DEF_GY;
    double p_damp = DEF_DAMP;
    double p_mass = REST_DENS * DEF_SPACING * DEF_SPACING;
    int    p_cols = DEF_COLS;
    int    p_rows = DEF_ROWS;
    int    p_nrand = 2000;
    bool   p_random = false;

    double HSQ = DEF_H * DEF_H;
    double CSIZE = DEF_H;      // cell size = H → 3×3 covers full kernel
    double K6 = 0.0;        // Poly6
    double KS = 0.0;        // Spiky grad
    double KV = 0.0;        // Visc lap

    // camera
    float  zoom = 1.0f;
    float  off_x = 0.0f;
    float  off_y = 0.0f;
    bool   panning = false;
    float  pan_ox = 0, pan_oy = 0; // mouse pos at pan start

    // mouse 
    bool   m_repel = false;
    V2     m_world = {};
    double m_radius = 45.0;
    double m_force = 6000.0;

    // timing
    double accum = 0.0;
    double last_t = 0.0;
    bool   t_init = false;

    

    SPHScreen() : Screen(Vector2i(BOX_W, BOX_H), "2D SPH Fluid") {

        Window* win = new Window(this, "Controls");
        win->setPosition(Vector2i(8, 8));
        win->setLayout(new GroupLayout());

        // Initialization
        new Label(win, "Init", "sans-bold");

        auto make_row = [&](Widget* parent) {
            Widget* row = new Widget(parent);
            row->setLayout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 4, 4));
            return row;
        };

        {
            Widget* row = make_row(win);
            new Label(row, "Random:", "sans");
            CheckBox* cb = new CheckBox(row, "");
            cb->setChecked(p_random);
            cb->setCallback([this](bool v) { p_random = v; });
        }

        auto add_int = [&](const std::string& name, int& val, int lo, int hi, int step = 1) {
            Widget* row = make_row(win);
            new Label(row, name + ":", "sans");
            auto* b = new IntBox<int>(row, val);
            b->setMinValue(lo); b->setMaxValue(hi);
            b->setValueIncrement(step); b->setEditable(true);
            b->setCallback([&val](int v) { val = v; });
        };

        add_int("Cols", p_cols, 3, 60, 1);
        add_int("Rows", p_rows, 3, 200, 5);
        add_int("Random", p_nrand, 50, 100000, 50);

        
        new Label(win, "Physics", "sans-bold");

        auto add_dbl = [&](const std::string& name, double& val,
            double lo, double hi, double step) {
                Widget* row = make_row(win);
                new Label(row, name + ":", "sans");
                auto* b = new FloatBox<double>(row, val);
                b->setMinValue(lo); b->setMaxValue(hi);
                b->setValueIncrement(step); b->setEditable(true);
                b->setCallback([&val](double v) { val = v; });
        };

        add_dbl("Spacing", p_spacing, 2.0, 20.0, 0.5);
        add_dbl("H", p_h, 4.0, 60.0, 1.0);
        add_dbl("dt", p_dt, 0.001, 0.02, 0.001);
        add_dbl("Gas const", p_gas, 1e3, 1e7, 1e3);
        add_dbl("Viscosity", p_visc, 1.0, 3000.0, 10.0);
        add_dbl("Gravity Y", p_gy, -50000.0, 0.0, 500.0);
        add_dbl("Global damp", p_damp, 0.95, 1.0, 0.001);

        Button* br = new Button(win, "Reset  [R]");
        br->setCallback([this]() { resetSimulation(); });

        Button* bp = new Button(win, "Pause  [Space]");
        bp->setCallback([this]() { running = !running; });

        lbl_fps = new Label(win, "FPS: --");
        lbl_count = new Label(win, "N = 0");

        new Label(win, "Mouse", "sans-bold");
        new Label(win, "RMB: repel", "sans");
        new Label(win, "LMB: pan", "sans");
        new Label(win, "Wheel: zoom", "sans");

        performLayout();
        resetSimulation();
    }

    // Reseting 
    void resetSimulation() {
        std::mt19937 rng(std::random_device{}());
        particles.clear();

        HSQ = p_h * p_h;
        CSIZE = p_h;
        const double h5 = std::pow(p_h, 5.0);
        const double h8 = std::pow(p_h, 8.0);
        K6 = 4.0 / (M_PI * h8);
        KS = -10.0 / (M_PI * h5);
        KV = 40.0 / (M_PI * h5);

        // Mass = REST_DENS * (spacing**2)
        p_mass = REST_DENS * p_spacing * p_spacing;

        const double xlo = EPS + 1, xhi = BOX_W - EPS - 1;
        const double ylo = EPS + 1, yhi = BOX_H - EPS - 1;

        if (p_random) {
            std::uniform_real_distribution<double> rx(xlo, xhi);
            std::uniform_real_distribution<double> ry(ylo, yhi);
            const int    target = std::clamp(p_nrand, 50, 20000);
            const double mind2 = 0.81 * p_spacing * p_spacing;
            const int    maxtry = target * 30;
            int tries = 0;
            while ((int)particles.size() < target && tries < maxtry) {
                ++tries;
                V2 pos{ rx(rng), ry(rng) };
                bool bad = false;
                for (const auto& q : particles)
                    if ((pos - q.pos).magSq() < mind2) { bad = true; break; }
                if (bad) continue;
                Particle p{}; p.pos = pos; p.rho = REST_DENS;
                particles.push_back(p);
            }
        }
        else {
            std::uniform_real_distribution<double> jit(
                -0.08 * p_spacing, +0.08 * p_spacing);
            const int cols = std::clamp(p_cols, 3, 80);
            const int rows = std::clamp(p_rows, 3, 300);
            const double sx = (BOX_W - cols * p_spacing) / 2.0;
            const double sy = ylo + p_spacing;
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c) {
                    V2 pos{ sx + c * p_spacing + jit(rng), sy + r * p_spacing + jit(rng) };
                    if (pos.x<xlo || pos.x>xhi || pos.y<ylo || pos.y>yhi) continue;
                    Particle p{}; p.pos = pos; p.rho = REST_DENS;
                    particles.push_back(p);
                }
        }

        accum = 0.0;
        t_init = false;

        if (lbl_count) {
            std::string mode = p_random ? " (random)"
                : " (grid " + std::to_string(std::clamp(p_cols, 3, 80))
                + "x" + std::to_string(std::clamp(p_rows, 3, 300)) + ")";
            lbl_count->setCaption("N = " + std::to_string(particles.size()) + mode);
        }
        std::cout << "Reset: " << particles.size()
            << " particles  H=" << p_h
            << "  mass=" << p_mass << "\n";
    }

    // Keyboard event

    virtual bool keyboardEvent(int key, int scan, int action, int mod) override {
        if (Screen::keyboardEvent(key, scan, action, mod)) return true;
        if (action == GLFW_PRESS) {
            if (key == GLFW_KEY_ESCAPE) { setVisible(false); return true; }
            if (key == GLFW_KEY_SPACE) { running = !running; return true; }
            if (key == GLFW_KEY_R) { resetSimulation();  return true; }
        }
        return false;
    }

    // Mouse event

    virtual bool scrollEvent(const Vector2i& p, const Vector2f& rel) override {
        if (Screen::scrollEvent(p, rel)) return true;
        zoom = std::clamp(zoom * (rel.y() > 0 ? 1.1f : 0.9f), 0.15f, 8.0f);
        return true;
    }

    virtual bool mouseButtonEvent(const Vector2i& p, int button, bool down, int mod) override {
        if (Screen::mouseButtonEvent(p, button, down, mod)) return true;
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            m_repel = down;
            if (down) m_world = screenToWorld(p);
            return true;
        }
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            panning = down;
            pan_ox = (float)p.x();
            pan_oy = (float)p.y();
            return true;
        }
        return Screen::mouseButtonEvent(p, button, down, mod);
    }

    virtual bool mouseMotionEvent(const Vector2i& p, const Vector2i& rel, int btn, int mod) override {
        if (Screen::mouseMotionEvent(p, rel, btn, mod)) return true;
        if (m_repel) {
            m_world = screenToWorld(p);
            return true;
        }
        if (panning) {
            off_x += (p.x() - pan_ox) / zoom;
            off_y -= (p.y() - pan_oy) / zoom;
            pan_ox = (float)p.x();
            pan_oy = (float)p.y();
            return true;
        }
        return Screen::mouseMotionEvent(p, rel, btn, mod);
    }

    // FPS counter

    virtual void draw(NVGcontext* ctx) override {
        static auto  t0 = std::chrono::high_resolution_clock::now();
        static int   cnt = 0;
        ++cnt;
        auto now = std::chrono::high_resolution_clock::now();
        long ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
        if (ms > 1000) {
            if (lbl_fps)
                lbl_fps->setCaption("FPS: " + std::to_string((int)(cnt * 1000.0 / ms)));
            cnt = 0; t0 = now;
        }
        Screen::draw(ctx);
    }

    // Main loop

    virtual void drawContents() override {
        double now = glfwGetTime();
        if (!t_init) { last_t = now; t_init = true; }
        double dt_wall = std::min(now - last_t, 0.05); // clamp to 50 ms
        last_t = now;

        accum += dt_wall;
        int steps = 0;
        while (accum >= p_dt && steps < DEF_MSTEPS) {
            if (running) updateSPH();
            accum -= p_dt;
            ++steps;
        }
        // Discard leftover that is too large (e.g. after a long pause)
        if (accum > p_dt * DEF_MSTEPS) accum = 0.0;

        drawParticles();
    }

private:

    V2 screenToWorld(const Vector2i& sp) const {
        double wx = (sp.x() - mSize.x() * 0.5) / zoom + mSize.x() * 0.5 - off_x;
        double wy = (sp.y() - mSize.y() * 0.5) / zoom + mSize.y() * 0.5 + off_y;
        return { wx, BOX_H - wy };
    }

    std::array<int, 2> cellKey(const V2& pos) const {
        return {
            (int)std::floor(pos.x / CSIZE),
            (int)std::floor(pos.y / CSIZE)
        };
    }

    // Iterate over all particles in the 3×3 neighbourhood of particle i
    template<typename CB>
    void forEachNeighbour(size_t i, const Grid& grid, CB&& cb) const {
        const auto key = cellKey(particles[i].pos);
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy) {
                std::array<int, 2> nk{key[0] + dx, key[1] + dy};
                Grid::const_accessor ca;
                if (grid.find(ca, nk))
                    for (int j : ca->second) cb(j);
            }
    }

    //  Colour: deep blue (rest) → cyan → white (fast)

    static void waterColor(double t, int& r, int& g, int& b) {
        t = std::clamp(t, 0.0, 1.0);
        if (t < 0.5) {
            double s = t * 2.0;
            r = (int)(10 + s * 90);
            g = (int)(80 + s * 120);
            b = (int)(200 + s * 55);
        }
        else {
            double s = (t - 0.5) * 2.0;
            r = (int)(100 + s * 130);
            g = (int)(200 + s * 45);
            b = 255;
        }
    }

    // Render

    void drawParticles() {
        NVGcontext* vg = mNVGContext;
        nvgBeginFrame(vg, mSize.x(), mSize.y(), 1.0f);

        nvgTranslate(vg, mSize.x() * 0.5f, mSize.y() * 0.5f);
        nvgScale(vg, zoom, zoom);
        nvgTranslate(vg, -mSize.x() * 0.5f + off_x,
            -mSize.y() * 0.5f - off_y);

        nvgBeginPath(vg);
        nvgRect(vg, 0, 0, BOX_W, BOX_H);
        nvgFillColor(vg, nvgRGBA(8, 12, 28, 255));
        nvgFill(vg);

        nvgBeginPath(vg);
        nvgRect(vg, 0, 0, BOX_W, BOX_H);
        nvgStrokeColor(vg, nvgRGBA(60, 100, 200, 140));
        nvgStrokeWidth(vg, 1.5f / zoom);
        nvgStroke(vg);

        // Particle visual radius
        const float vr = (float)(p_h * 0.45);

        double max_spd_sq = 1.0;
        for (const auto& pt : particles)
            max_spd_sq = std::max(max_spd_sq, pt.vel.magSq());
        const double inv_spd = 1.0 / std::sqrt(max_spd_sq);

        for (const auto& pt : particles) {
            const double t = std::clamp(pt.vel.mag() * inv_spd, 0.0, 1.0);
            int cr, cg, cb; waterColor(t, cr, cg, cb);

            const float px = (float)pt.pos.x;
            const float py = (float)(BOX_H - pt.pos.y);

            nvgBeginPath(vg);
            nvgCircle(vg, px, py, vr);
            NVGpaint paint = nvgRadialGradient(vg,
                px, py,
                vr * 0.15f, vr * 1.1f,
                nvgRGBA(cr, cg, cb, 220),
                nvgRGBA(cr / 5, cg / 5, cb / 3, 0));
            nvgFillPaint(vg, paint);
            nvgFill(vg);
        }

        if (m_repel) {
            nvgBeginPath(vg);
            nvgCircle(vg, (float)m_world.x, (float)(BOX_H - m_world.y), (float)m_radius);
            nvgStrokeColor(vg, nvgRGBA(255, 255, 255, 90));
            nvgStrokeWidth(vg, 1.0f / zoom);
            nvgStroke(vg);
        }

        nvgResetTransform(vg);
        nvgFontSize(vg, 13.0f);
        nvgFontFace(vg, "sans");
        nvgFillColor(vg, nvgRGBA(160, 190, 255, 200));
        nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_TOP);
        nvgText(vg, 10, 10, "SPH 2D  |  SPACE=Pause  R=Reset  ESC=Exit", nullptr);
        nvgText(vg, 10, 26, "RMB=repel  LMB=pan  Wheel=zoom", nullptr);

        nvgEndFrame(vg);
    }

    // Physics loop

    void updateSPH() {
        const size_t N = particles.size();
        if (N == 0) return;

        const double h = p_h;
        const double hsq = HSQ;
        const double mass = p_mass;
        const double gas = p_gas;
        const double visc = p_visc;
        const double gy = p_gy;
        const double damp = p_damp;
        const double dt = p_dt;

        // Build spatial grid 
        Grid grid;

        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, N),
            [&](const auto& rng) {
                for (size_t i = rng.begin(); i != rng.end(); ++i) {
                    auto key = cellKey(particles[i].pos);
                    Grid::accessor acc;
                    if (grid.insert(acc, key)) acc->second.reserve(16);
                    acc->second.push_back((int)i);
                }
            });

        // Density + Pressure 
        // Tait EOS: p = B((rho/rho0)^7 - 1)

        constexpr double GAMMA = 7.0;
        const double B = gas * REST_DENS / GAMMA;

        const double self_hd3 = hsq * hsq * hsq;
        const double self_rho = mass * K6 * self_hd3;

        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, N),
            [&](const auto& rng) {
                for (size_t i = rng.begin(); i != rng.end(); ++i) {
                    auto& pi = particles[i];
                    double rho = self_rho; 

                    forEachNeighbour(i, grid, [&](int j) {
                        if ((size_t)j == i) return;
                        const double dsq = (pi.pos - particles[j].pos).magSq();
                        if (dsq < hsq) {
                            const double hd = hsq - dsq;
                            rho += mass * K6 * hd * hd * hd;
                        }
                        });

                    pi.rho = std::max(rho, REST_DENS * 0.01);

                    // Tait EOS — replace std::pow(x,7) with repeated multiply (6× faster)
                    const double ratio = pi.rho / REST_DENS;
                    const double r2 = ratio * ratio;
                    const double r4 = r2 * r2;
                    const double r7 = r4 * r2 * ratio;
                    pi.p = B * (r7 - 1.0);
                }
            });

        // Forces (pressure + viscosity + gravity) 
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, N),
            [&](const auto& rng) {
                for (size_t i = rng.begin(); i != rng.end(); ++i) {
                    auto& pi = particles[i];
                    V2 fp{}, fv{};

                    forEachNeighbour(i, grid, [&](int k) {
                        if ((size_t)k == i) return;
                        const auto& pj = particles[k];
                        const double dsq = (pi.pos - pj.pos).magSq();
                        const double r = std::sqrt(dsq);
                        if (r >= h || r < 1e-5) return;

                        const V2     dir = (pi.pos - pj.pos) / r;
                        const double hr = h - r;

                        // Pressure 
                        const double pk = KS * hr * hr;
                        const double sp = mass * (pi.p / (pi.rho * pi.rho) + pj.p / (pj.rho * pj.rho));
                        fp += dir * (-sp * pk);

                        // Viscosity
                        const double vk = KV * hr;
                        fv += (pj.vel - pi.vel) * (visc * mass * vk / pj.rho);
                        });

                    // Gravity acts on mass 
                    pi.force = fp + fv + V2{0.0, gy* pi.rho};
                }
            });

        // Integrate + boundaries 
        const double vmax = h / dt * 0.4; // velocity cap: CFL-based

        for (auto& p : particles) {
            V2 accel = p.force / p.rho;

            // Mouse repulsion
            if (m_repel) {
                V2 d = p.pos - m_world;
                double dist = d.mag();
                if (dist < m_radius && dist > 1e-5)
                    accel += d.norm() * (m_force * (1.0 - dist / m_radius));
            }

            // Integrate velocity
            p.vel += accel * dt;

            // Global damping — kills bulk sloshing that viscosity cannot damp
            p.vel *= damp;

            // Velocity cap — prevents CFL blow-up from transient large forces
            const double spd = p.vel.mag();
            if (spd > vmax) p.vel = p.vel * (vmax / spd);

            // Integrate position
            p.pos += p.vel * dt;

            // Wall collisions
            auto bounce = [&](double& pos, double& vel, double lo, double hi) {
                if (pos < lo) { vel = std::abs(vel) * WALL_DAMP; pos = lo + 1e-4; }
                if (pos > hi) { vel = -std::abs(vel) * WALL_DAMP; pos = hi - 1e-4; }
            };
            bounce(p.pos.x, p.vel.x, EPS, BOX_W - EPS);
            bounce(p.pos.y, p.vel.y, EPS, BOX_H - EPS);
        }
    }
};



int main(int argc, char** argv) {
    try {
        nanogui::init();
        {
            ref<SPHScreen> screen = new SPHScreen();
            screen->drawAll();
            screen->setVisible(true);
            nanogui::mainloop();
        }
        nanogui::shutdown();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
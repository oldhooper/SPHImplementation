#define _USE_MATH_DEFINES
#define NOMINMAX          // prevent Windows near/far/min/max macros

#include <nanogui/nanogui.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>
#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <string>

using namespace nanogui;



constexpr int    BOX_W = 1200;
constexpr int    BOX_H = 800;
constexpr int    PANEL_W = 210;  // width of each control panel column
constexpr int    WIN_W = BOX_W + PANEL_W * 2 + 24; // box + two panels + margins
constexpr int    WIN_H = BOX_H;

// Defaults 
constexpr int    DEF_COLS = 100;
constexpr int    DEF_ROWS = 100;
constexpr double DEF_SPACING = 10.0;
constexpr double DEF_H = 6.0;
constexpr double DEF_DT = 0.0008;
constexpr int    DEF_MSTEPS = 4;
constexpr double REST_DENS = 1000.0;
constexpr double DEF_GAS = 8e4;
constexpr double DEF_VISC = 50.0;
constexpr double DEF_GY = -9800.0;
constexpr double DEF_DAMP = 0.992;
constexpr double EPS = 2.0;
constexpr double WALL_DAMP = 0.1;

// Default cone obstacle params
constexpr double DEF_CONE_CX_PCT = 0.50;
constexpr double DEF_CONE_TY_PCT = 0.82;
constexpr double DEF_CONE_HW = 55.0;
constexpr double DEF_CONE_HT = 90.0;

// v2

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

// particle

struct Particle {
    V2     pos{}, vel{}, force{};
    double rho = REST_DENS;
    double p = 0.0;
    int    neighbours = 0;   // count of particles within kernel radius H (excluding self)
};

// ============================================================
//  Rigid obstacle: upward-pointing triangle (apex on top)
//
//  Boundary condition: penalty-force method.
//  For each particle that penetrates the cone surface by depth d,
//  we apply an outward impulse:  F = STIFFNESS * d * outward_normal
//
//  The cone is defined by 3 half-planes (edges).  A point is
//  inside the solid if it satisfies all 3 inequalities.
//  We resolve the collision against the closest surface only.
//
//  Vertices (world-space, Y-up):
//      apex  = top centre
//      left  = bottom-left
//      right = bottom-right
//
//  Winding (CCW):  left → right → apex
//  Interior is to the LEFT of every directed edge.
// ============================================================

struct ConeObstacle {
    V2     apex, left, right;  // world-space corners
    double stiffness = 0.0;

    // cx        = centre x
    // tip_y     = y of the apex (TOP of the triangle)
    // half_w    = half the base width
    // height    = vertical extent (base is at tip_y - height)
    void build(double cx, double tip_y, double half_w, double height, double stiff) {
        apex = { cx,          tip_y };
        left = { cx - half_w, tip_y - height };
        right = { cx + half_w, tip_y - height };
        stiffness = stiff;
    }

    // Signed distance from P to directed line A→B.
    // Positive = left side of AB, negative = right side.
    static double sdLine(const V2& p, const V2& a, const V2& b) {
        V2 ab{ b.x - a.x, b.y - a.y };
        double len = ab.mag();
        if (len < 1e-9) return 0.0;
        V2 n{ -ab.y / len, ab.x / len };          // left-normal
        return (p.x - a.x) * n.x + (p.y - a.y) * n.y;
    }

    // Returns true if P is inside the solid cone.
    // Fills push_normal (unit, outward) and depth (>0).
    bool collide(const V2& pos, V2& push_normal, double& depth) const {
        // CCW winding: left → right → apex
        // Interior is to the LEFT of each directed edge → sdLine > 0
        //
        //   BOTTOM EDGE (left→right) :  interior is ABOVE   → sdLine > 0
        //   RIGHT WALL  (right→apex) :  interior is to LEFT → sdLine > 0
        //   LEFT WALL   (apex→left)  :  interior is to LEFT → sdLine > 0
        double d_bot = sdLine(pos, left, right);  // pos = inside (above base)
        double d_rw = sdLine(pos, right, apex);   // pos = inside (left of right wall)
        double d_lw = sdLine(pos, apex, left);   // pos = inside (left of left wall)

        // Particle is inside solid only when all three are positive
        if (d_bot <= 0.0 || d_rw <= 0.0 || d_lw <= 0.0) return false;

        //push away from interior
        struct Wall { double d; V2 a, b; };
        Wall walls[3] = {
            { d_bot, left,  right },  // bottom edge
            { d_rw,  right, apex  },  // right wall
            { d_lw,  apex,  left  },  // left wall
        };

        int best = 0;
        for (int i = 1; i < 3; ++i)
            if (walls[i].d < walls[best].d) best = i;

        depth = walls[best].d;

        const Wall& w = walls[best];
        V2 ab{ w.b.x - w.a.x, w.b.y - w.a.y };
        double len = ab.mag();
        V2 ln{ -ab.y / len, ab.x / len };   // left-normal (points INTO solid)
        push_normal = { -ln.x, -ln.y };      // outward = negate
        return true;
    }

    void draw(NVGcontext* vg) const {
        // Filled body — warm amber/orange so it reads clearly on dark background
        nvgBeginPath(vg);
        nvgMoveTo(vg, (float)apex.x, (float)(BOX_H - apex.y));
        nvgLineTo(vg, (float)right.x, (float)(BOX_H - right.y));
        nvgLineTo(vg, (float)left.x, (float)(BOX_H - left.y));
        nvgClosePath(vg);
        nvgFillColor(vg, nvgRGBA(160, 80, 20, 220));
        nvgFill(vg);
        // Bright outline
        nvgStrokeColor(vg, nvgRGBA(255, 170, 60, 255));
        nvgStrokeWidth(vg, 2.0f);
        nvgStroke(vg);
        // Apex dot
        nvgBeginPath(vg);
        nvgCircle(vg, (float)apex.x, (float)(BOX_H - apex.y), 3.5f);
        nvgFillColor(vg, nvgRGBA(255, 220, 100, 255));
        nvgFill(vg);
    }
};

// ============================================================
//  Main simulation screen

class SPHScreen : public Screen {
public:
    std::vector<Particle> particles;
    bool running = true;

    Label* lbl_fps = nullptr;
    Label* lbl_count = nullptr;

    // ---- live physics params ----
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
    int    p_nrand = 300;
    bool   p_random = false;

    // derived kernel constants (recomputed on reset) 
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

    // mouse interaction 
    bool   m_repel = false;
    V2     m_world = {};
    double m_radius = 45.0;
    double m_force = 6000.0;

    // timing (not static — survives pause/resume) 
    double accum = 0.0;
    double last_t = 0.0;
    bool   t_init = false;

    // cone obstacle 
    ConeObstacle cone;
    double cone_cx_pct = DEF_CONE_CX_PCT;
    double cone_ty_pct = DEF_CONE_TY_PCT;
    double cone_hw = DEF_CONE_HW;
    double cone_ht = DEF_CONE_HT;
    bool   cone_enable = true;

    // metrics labels (right panel)
    Label* lbl_avg_rho = nullptr;
    Label* lbl_avg_nb = nullptr;
    Label* lbl_max_vel = nullptr;

    SPHScreen() : Screen(Vector2i(WIN_W, WIN_H), "2D SPH Fluid") {

        // Helper: horizontal row inside a panel
        auto make_row = [&](Widget* parent) {
            Widget* row = new Widget(parent);
            row->setLayout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 4, 4));
            return row;
        };

        // Helper: double spinner
        auto add_dbl = [&](Widget* panel, const std::string& name, double& val,
            double lo, double hi, double step) {
                Widget* row = make_row(panel);
                new Label(row, name + ":", "sans");
                auto* b = new FloatBox<double>(row, val);
                b->setMinValue(lo); b->setMaxValue(hi);
                b->setValueIncrement(step); b->setEditable(true);
                b->setCallback([&val](double v) { val = v; });
        };

        // Helper: int spinner
        auto add_int = [&](Widget* panel, const std::string& name, int& val,
            int lo, int hi, int step = 1) {
                Widget* row = make_row(panel);
                new Label(row, name + ":", "sans");
                auto* b = new IntBox<int>(row, val);
                b->setMinValue(lo); b->setMaxValue(hi);
                b->setValueIncrement(step); b->setEditable(true);
                b->setCallback([&val](int v) { val = v; });
        };

        // ================================================================
        //  LEFT PANEL — Model parameters
        // ================================================================
        Window* left = new Window(this, "Model");
        left->setPosition(Vector2i(BOX_W + 4, 4));
        left->setLayout(new GroupLayout());

        new Label(left, "Init", "sans-bold");
        {
            Widget* row = make_row(left);
            new Label(row, "Random:", "sans");
            CheckBox* cb = new CheckBox(row, "");
            cb->setChecked(p_random);
            cb->setCallback([this](bool v) { p_random = v; });
        }
        add_int(left, "Cols", p_cols, 3, 1000, 1);
        add_int(left, "Rows", p_rows, 3, 1000, 5);
        add_int(left, "RandN", p_nrand, 50, 50000, 100);

        new Label(left, "Physics", "sans-bold");
        add_dbl(left, "Spacing", p_spacing, 2.0, 20.0, 0.5);
        add_dbl(left, "H", p_h, 1.0, 60.0, 0.5);
        add_dbl(left, "dt", p_dt, 0.00001, 0.02, 0.0001);
        add_dbl(left, "Gas", p_gas, 1e3, 1e7, 1e3);
        add_dbl(left, "Visc", p_visc, 1.0, 3000.0, 10.0);
        add_dbl(left, "Grav Y", p_gy, -50000.0, 0.0, 500.0);
        add_dbl(left, "Damp", p_damp, 0.95, 1.0, 0.001);

        Button* br = new Button(left, "Reset  [R]");
        br->setCallback([this]() { resetSimulation(); });
        Button* bp = new Button(left, "Pause  [Space]");
        bp->setCallback([this]() { running = !running; });

        lbl_fps = new Label(left, "FPS: --");
        lbl_count = new Label(left, "N = 0");

        new Label(left, "Mouse", "sans-bold");
        new Label(left, "RMB: repel", "sans");
        new Label(left, "LMB: pan", "sans");
        new Label(left, "Wheel: zoom", "sans");

        // ================================================================
        //  RIGHT PANEL — Cone obstacle + Metrics
        // ================================================================
        Window* right = new Window(this, "Cone / Info");
        right->setPosition(Vector2i(BOX_W + PANEL_W + 12, 4));
        right->setLayout(new GroupLayout());

        new Label(right, "Cone Obstacle", "sans-bold");
        {
            Widget* row = make_row(right);
            new Label(row, "Enabled:", "sans");
            CheckBox* cb = new CheckBox(row, "");
            cb->setChecked(cone_enable);
            cb->setCallback([this](bool v) { cone_enable = v; rebuildCone(); });
        }
        add_dbl(right, "X %", cone_cx_pct, 0.05, 0.95, 0.05);
        add_dbl(right, "Y %", cone_ty_pct, 0.05, 0.98, 0.05);
        add_dbl(right, "HalfW", cone_hw, 5.0, 140.0, 5.0);
        add_dbl(right, "Height", cone_ht, 10.0, 400.0, 5.0);

        Button* bc = new Button(right, "Rebuild Cone");
        bc->setCallback([this]() { rebuildCone(); });

        new Label(right, "Live Metrics", "sans-bold");
        lbl_avg_rho = new Label(right, "Avg density: --");
        lbl_avg_nb = new Label(right, "Avg neighbours: --");
        lbl_max_vel = new Label(right, "Max velocity: --");

        new Label(right, "Limits", "sans-bold");
        new Label(right, "~5k  @60fps smooth", "sans");
        new Label(right, "~15k @30fps OK", "sans");
        new Label(right, "~30k+ gets heavy", "sans");

        performLayout();
        resetSimulation();
    }

    // Reset

    void resetSimulation() {
        std::mt19937 rng(std::random_device{}());
        particles.clear();

        // Recompute kernel constants
        HSQ = p_h * p_h;
        CSIZE = p_h;
        const double h5 = std::pow(p_h, 5.0);
        const double h8 = std::pow(p_h, 8.0);
        K6 = 4.0 / (M_PI * h8);
        KS = -10.0 / (M_PI * h5);
        KV = 40.0 / (M_PI * h5);

        // Mass = REST_DENS × spacing² (2D)
        p_mass = REST_DENS * p_spacing * p_spacing;

        const double xlo = EPS + 1, xhi = BOX_W - EPS - 1;
        const double ylo = EPS + 1, yhi = BOX_H - EPS - 1;

        if (p_random) {
            std::uniform_real_distribution<double> rx(xlo, xhi);
            std::uniform_real_distribution<double> ry(ylo, yhi);
            const int    target = std::clamp(p_nrand, 50, 50000);
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
            const int cols = std::clamp(p_cols, 3, 1000);
            const int rows = std::clamp(p_rows, 3, 1000);
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

        rebuildCone();

        if (lbl_count) {
            std::string mode = p_random ? " (random)"
                : " (grid " + std::to_string(std::clamp(p_cols, 3, 120))
                + "x" + std::to_string(std::clamp(p_rows, 3, 800)) + ")";
            lbl_count->setCaption("N = " + std::to_string(particles.size()) + mode);
        }
        std::cout << "Reset: " << particles.size()
            << " particles  H=" << p_h
            << "  mass=" << p_mass << "\n";
    }

    void rebuildCone() {
        if (cone_enable) {
            cone.build(
                BOX_W * cone_cx_pct,
                BOX_H * cone_ty_pct,
                cone_hw,
                cone_ht,
                p_gas * REST_DENS * 0.8
            );
        }
        else {
            cone.build(-9999.0, -9999.0, 0.1, 0.1, 0.0);
        }
    }

    // Keyboard

    virtual bool keyboardEvent(int key, int scan, int action, int mod) override {
        if (Screen::keyboardEvent(key, scan, action, mod)) return true;
        if (action == GLFW_PRESS) {
            if (key == GLFW_KEY_ESCAPE) { setVisible(false); return true; }
            if (key == GLFW_KEY_SPACE) { running = !running; return true; }
            if (key == GLFW_KEY_R) { resetSimulation();  return true; }
        }
        return false;
    }

    // Mouse

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

    // FPS

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
        const float box_cx = BOX_W * 0.5f;
        const float box_cy = BOX_H * 0.5f;
        double wx = (sp.x() - box_cx) / zoom + box_cx - off_x;
        double wy = (sp.y() - box_cy) / zoom + box_cy + off_y;
        return { wx, BOX_H - wy };
    }

    static void neighbourColor(int nb, int& r, int& g, int& b) {
        // Map neighbour count to [0,1]: saturate at 35
        const double t = std::clamp(nb / 35.0, 0.0, 1.0);

        if (t < 0.25) {
            // 0-25%: black → deep purple
            double s = t / 0.25;
            r = (int)(0 + s * 60);
            g = (int)(0 + s * 10);
            b = (int)(30 + s * 100);
        }
        else if (t < 0.5) {
            // 25-50%: deep purple → deep blue
            double s = (t - 0.25) / 0.25;
            r = (int)(60 - s * 50);
            g = (int)(10 + s * 60);
            b = (int)(130 + s * 120);
        }
        else if (t < 0.75) {
            // 50-75%: deep blue → cyan
            double s = (t - 0.5) / 0.25;
            r = (int)(10 + s * 20);
            g = (int)(70 + s * 160);
            b = (int)(250);
        }
        else {
            // 75-100%: cyan → bright white/green
            double s = (t - 0.75) / 0.25;
            r = (int)(30 + s * 220);
            g = (int)(230 + s * 25);
            b = (int)(250 - s * 120);
        }
    }

    // render

    void drawParticles() {
        NVGcontext* vg = mNVGContext;
        nvgBeginFrame(vg, mSize.x(), mSize.y(), 1.0f);

        // Camera transform — centre zoom on the BOX area (left side of window)
        const float box_cx = BOX_W * 0.5f;
        const float box_cy = BOX_H * 0.5f;
        nvgTranslate(vg, box_cx, box_cy);
        nvgScale(vg, zoom, zoom);
        nvgTranslate(vg, -box_cx + off_x,
            -box_cy - off_y);

        // Background
        nvgBeginPath(vg);
        nvgRect(vg, 0, 0, BOX_W, BOX_H);
        nvgFillColor(vg, nvgRGBA(8, 12, 28, 105));
        nvgFill(vg);

        // Container border
        nvgBeginPath(vg);
        nvgRect(vg, 0, 0, BOX_W, BOX_H);
        nvgStrokeColor(vg, nvgRGBA(60, 100, 200, 140));
        nvgStrokeWidth(vg, 1.5f / zoom);
        nvgStroke(vg);

        // Cone obstacle
        if (cone_enable) cone.draw(vg);

        // Particle visual radius: H×0.45 so neighbours overlap → fluid blob
        const float vr = (float)(p_h * 0.45);

        for (const auto& pt : particles) {
            int cr, cg, cb; neighbourColor(pt.neighbours, cr, cg, cb);

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

        // Mouse repel cursor
        if (m_repel) {
            nvgBeginPath(vg);
            nvgCircle(vg, (float)m_world.x, (float)(BOX_H - m_world.y), (float)m_radius);
            nvgStrokeColor(vg, nvgRGBA(255, 255, 255, 90));
            nvgStrokeWidth(vg, 1.0f / zoom);
            nvgStroke(vg);
        }

        // HUD (screen space — reset transform first)
        nvgResetTransform(vg);
        nvgFontSize(vg, 13.0f);
        nvgFontFace(vg, "sans");
        nvgFillColor(vg, nvgRGBA(160, 190, 255, 200));
        nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_TOP);
        nvgText(vg, 10, 10, "SPH 2D  |  SPACE=Pause  R=Reset  ESC=Exit", nullptr);
        nvgText(vg, 10, 26, "RMB=repel  LMB=pan  Wheel=zoom", nullptr);

        // Update live metrics (every frame, cheap for display)
        {
            const int N = (int)particles.size();
            if (N > 0) {
                double sum_rho = 0, sum_nb = 0, mx_vel = 0;
                for (const auto& pt : particles) {
                    sum_rho += pt.rho;
                    sum_nb += pt.neighbours;
                    double v = pt.vel.mag();
                    if (v > mx_vel) mx_vel = v;
                }
                if (lbl_avg_rho)
                    lbl_avg_rho->setCaption("Avg density: " + std::to_string((int)(sum_rho / N)));
                if (lbl_avg_nb)
                    lbl_avg_nb->setCaption("Avg neighbours: " + std::to_string((int)(sum_nb / N)));
                if (lbl_max_vel)
                    lbl_max_vel->setCaption("Max vel: " + std::to_string((int)mx_vel));
            }
        }

        nvgEndFrame(vg);
    }

    //  Physics

    std::vector<uint32_t> sorted_ids;
    std::vector<uint32_t> particle_cell;   // cell index for each particle
    std::vector<uint32_t> cell_start;
    std::vector<uint32_t> cell_count;
    int grid_nx = 0, grid_ny = 0;
    int grid_total = 0;

    void rebuildGrid(int N) {
        // Grid dimensions from box size and cell size
        grid_nx = (int)std::ceil((double)BOX_W / CSIZE) + 2;
        grid_ny = (int)std::ceil((double)BOX_H / CSIZE) + 2;
        grid_total = grid_nx * grid_ny;

        sorted_ids.resize(N);
        particle_cell.resize(N);
        cell_start.assign(grid_total, UINT32_MAX);
        cell_count.assign(grid_total, 0);

        // Compute cell index for every particle
        for (int i = 0; i < N; ++i) {
            int cx = std::clamp((int)std::floor(particles[i].pos.x / CSIZE), 0, grid_nx - 1);
            int cy = std::clamp((int)std::floor(particles[i].pos.y / CSIZE), 0, grid_ny - 1);
            uint32_t c = (uint32_t)(cx + cy * grid_nx);
            particle_cell[i] = c;
            cell_count[c]++;
        }

        // Exclusive prefix sum → cell_start
        uint32_t acc = 0;
        for (int c = 0; c < grid_total; ++c) {
            if (cell_count[c] > 0) {
                cell_start[c] = acc;
                acc += cell_count[c];
            }
        }

        // Fill sorted_ids (use cell_start as a write cursor, then restore)
        std::vector<uint32_t> cursor(cell_start); // copy
        for (int i = 0; i < N; ++i) {
            uint32_t c = particle_cell[i];
            sorted_ids[cursor[c]++] = (uint32_t)i;
        }
    }

    // Iterate over all particles in the 3×3 neighbourhood of a cell (cx,cy)
    template<typename CB>
    void forEachNeighbourFlat(int cx, int cy, CB&& cb) const {
        for (int dy = -1; dy <= 1; ++dy) {
            int ny = cy + dy;
            if (ny < 0 || ny >= grid_ny) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int nx2 = cx + dx;
                if (nx2 < 0 || nx2 >= grid_nx) continue;
                uint32_t c = (uint32_t)(nx2 + ny * grid_nx);
                uint32_t start = cell_start[c];
                if (start == UINT32_MAX) continue;
                uint32_t end = start + cell_count[c];
                for (uint32_t k = start; k < end; ++k)
                    cb(sorted_ids[k]);
            }
        }
    }

    void updateSPH() {
        const int N = (int)particles.size();
        if (N == 0) return;

        // Cache params locally — avoids member pointer chasing in hot loops
        const double h = p_h;
        const double hsq = HSQ;
        const double mass = p_mass;
        const double gas = p_gas;
        const double visc = p_visc;
        const double gy = p_gy;
        const double damp = p_damp;
        const double dt = p_dt;

        // Build flat grid (lock-free, cache-friendly) 
        rebuildGrid(N);

        constexpr double GAMMA = 7.0;
        const double B = gas * REST_DENS / GAMMA;
        const double self_rho = mass * K6 * hsq * hsq * hsq;
        const double vmax = h / dt * 0.4;

        // Density + Pressure (parallel, read-only grid) 
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, N, 64),
            [&](const auto& rng) {
                for (int i = rng.begin(); i != rng.end(); ++i) {
                    auto& pi = particles[i];
                    double rho = self_rho;
                    int    nb = 0;

                    int cx = std::clamp((int)std::floor(pi.pos.x / CSIZE), 0, grid_nx - 1);
                    int cy = std::clamp((int)std::floor(pi.pos.y / CSIZE), 0, grid_ny - 1);

                    forEachNeighbourFlat(cx, cy, [&](uint32_t j) {
                        if ((int)j == i) return;
                        const double dsq = (pi.pos - particles[j].pos).magSq();
                        if (dsq < hsq) {
                            const double hd = hsq - dsq;
                            rho += mass * K6 * hd * hd * hd;
                            ++nb;
                        }
                        });

                    pi.rho = std::max(rho, REST_DENS * 0.01);
                    pi.neighbours = nb;

                    // Tait EOS — repeated multiply instead of pow(x,7)
                    const double ratio = pi.rho / REST_DENS;
                    const double r2 = ratio * ratio, r4 = r2 * r2;
                    pi.p = B * (r4 * r2 * ratio - 1.0);
                }
            });

        // Forces (parallel, read-only grid + density)
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, N, 64),
            [&](const auto& rng) {
                for (int i = rng.begin(); i != rng.end(); ++i) {
                    auto& pi = particles[i];
                    V2 fp{}, fv{};

                    int cx = std::clamp((int)std::floor(pi.pos.x / CSIZE), 0, grid_nx - 1);
                    int cy = std::clamp((int)std::floor(pi.pos.y / CSIZE), 0, grid_ny - 1);

                    forEachNeighbourFlat(cx, cy, [&](uint32_t k) {
                        if ((int)k == i) return;
                        const auto& pj = particles[k];
                        const double dsq = (pi.pos - pj.pos).magSq();
                        if (dsq >= hsq || dsq < 1e-10) return;
                        const double r = std::sqrt(dsq);
                        if (r >= h) return;
                        const V2     dir = (pi.pos - pj.pos) / r;
                        const double hr = h - r;

                        // Pressure — symmetric form (conserves momentum)
                        const double pk = KS * hr * hr;
                        const double sp = mass * (pi.p / (pi.rho * pi.rho) + pj.p / (pj.rho * pj.rho));
                        fp += dir * (-sp * pk);

                        // Viscosity
                        fv += (pj.vel - pi.vel) * (visc * mass * KV * hr / pj.rho);
                        });

                    pi.force = fp + fv + V2{0.0, gy* pi.rho};
                }
            });

        // Integrate + boundaries (serial — each particle independent) 
        const double cone_stiff = p_gas * REST_DENS * 0.8;

        for (auto& p : particles) {
            V2 accel = p.force / p.rho;

            // Mouse repulsion
            if (m_repel) {
                V2 d = p.pos - m_world;
                double dist = d.mag();
                if (dist < m_radius && dist > 1e-5)
                    accel += d.norm() * (m_force * (1.0 - dist / m_radius));
            }

            // Cone penalty force
            if (cone_enable) {
                V2 push_n; double depth;
                if (cone.collide(p.pos, push_n, depth)) {
                    accel += push_n * (cone_stiff * depth / p.rho);
                    double vn = p.vel.x * (-push_n.x) + p.vel.y * (-push_n.y);
                    if (vn > 0.0) {
                        p.vel.x += push_n.x * vn * (1.0 + WALL_DAMP);
                        p.vel.y += push_n.y * vn * (1.0 + WALL_DAMP);
                    }
                }
            }

            p.vel += accel * dt;
            p.vel *= damp;

            // Velocity cap
            const double spd = p.vel.mag();
            if (spd > vmax) p.vel = p.vel * (vmax / spd);

            p.pos += p.vel * dt;

            // Wall bounce
            auto bounce = [](double& pos, double& vel, double lo, double hi) {
                if (pos < lo) { vel = std::abs(vel) * WALL_DAMP; pos = lo + 1e-4; }
                if (pos > hi) { vel = -std::abs(vel) * WALL_DAMP; pos = hi - 1e-4; }
            };
            bounce(p.pos.x, p.vel.x, EPS, BOX_W - EPS);
            bounce(p.pos.y, p.vel.y, EPS, BOX_H - EPS);
        }
    }
};

//  Entry point

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
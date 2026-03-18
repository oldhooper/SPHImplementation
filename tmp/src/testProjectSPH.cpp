#define _USE_MATH_DEFINES

//#include "C:\Users\admin\dev\SPHImplementation\tmp\libs\nanogui\ext\glad\include\glad\glad.h"
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

using namespace nanogui;


//
// Физика работает, но частицы заходят друг на дурга и не растекаются
//

// ==================== КОНСТАНТЫ ====================

constexpr int   BOX_WIDTH = 300;
constexpr int   BOX_HEIGHT = 600;

constexpr int   COLUMNS_NUM = 17;
constexpr int   ROWS_NUM = 100;

constexpr double SPACING = 2.0;         // фиксированное расстояние между частицами 

constexpr double H = 20.0;           // было 0.1
constexpr double HSQ = H * H;


//Можно потестировать ограничение на шаги -> возможно повлияет на фпс и плавность анимации
constexpr double DT = 0.001;            // чем меньше, тем точнее и медленнее
constexpr int MAX_STEPS = 10;            // ограничение шагов на кадр drawContext

constexpr double REST_DENS = 1000.0;
constexpr double GAS_CONST = 1000000.0;        // тестировал с 12к, 80к, 200к, 1m на высоких значениях - замедление программы
constexpr double VISC = 30.0;         // уменьшил с 20 до 3
constexpr double MASS = 50.0;        // (было 25000) расчитывается по формуле rest_density*spacing**2, для стабильности понизим 
constexpr double GX = 0.0, GY = -1000.0; // было 30 



constexpr double EPS = H; //отвечает за отступ от границ резервуара
constexpr double CELL_SIZE = H*2;             // теперь 2H вместо H, для того чтобы не терялись соседи
constexpr double BOUND_DAMPING = 0.99;       // было 0.3


// 2D Kernels (оставляем как было)
constexpr double H_POW_5 = H * H * H * H * H;
constexpr double H_POW_8 = H_POW_5 * H * H * H;
//constexpr double H_POW_6 = H_POW_5 * H;

const double POLY6 = 4.0 / (M_PI * H_POW_8); 
const double SPIKY_GRAD = -10.0 / (M_PI * H_POW_5);        // изменил коэф с 30 на -10
const double VISC_LAP = 40.0 / (M_PI * H_POW_5);            // изменил коэф с 45 на 40 и степень с 6 на 5



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
        std::uniform_real_distribution<double> jitter(-0.12, 0.12);

        // Create a dam break setup - particles in the left part of the container
        int cols = COLUMNS_NUM;      // было 20
        int rows = ROWS_NUM;     // было 90
        particles.resize(cols * rows);
       
       

        // Fill particles in a rectangular block on the left
        double spacing = SPACING;
        double start_x = (BOX_WIDTH - cols * spacing) / 2;            // отступ от левой стенки
        double start_y = 0.0;

        int idx = 0;
        for (int i = 0; i < cols; ++i) {          // i — по ширине (X)
            for (int j = 0; j < rows; ++j) {      // j — по высоте (Y)
                particles[idx].pos = Vector2{
                    start_x + i * spacing + jitter(gen),
                    start_y + j * spacing + jitter(gen)   // начинаем чуть выше дна
                };
                ++idx;
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

    //Отвечает за перезапуск симуляции 
    void resetSimulation() {
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> jitter(-0.12, 0.12);

        int cols = COLUMNS_NUM;      // было 20
        int rows = ROWS_NUM;     // было 90
        double spacing = SPACING;

        double start_x = (BOX_WIDTH - cols * spacing) / 2;
        double start_y = 0.0;

        particles.resize(cols * rows);

        for (int i = 0; i < cols; ++i) {
            for (int j = 0; j < rows; ++j) {
                size_t idx = static_cast<size_t>(i * rows + j);
                particles[idx].pos = Vector2{
                    start_x + i * spacing + jitter(gen),
                    start_y + j * spacing + jitter(gen)
                };
                particles[idx].vel = Vector2{ 0.0, 0.0 };
                particles[idx].force = Vector2{ 0.0, 0.0 };
                particles[idx].rho = REST_DENS;
                particles[idx].p = 0.0;
            }
        }
        needs_redraw = true;
    }

    //Хэндлер нажатий на клавиатуру
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

    //Отрисовщик сцены
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

    //Отрисовщик частиц
    virtual void drawContents() override {
        static double last_time = glfwGetTime();
        double current_time = glfwGetTime();
        double delta_time = current_time - last_time;
        last_time = current_time;

        static double accumulator = 0.0;
        accumulator += delta_time;

        const double step = DT;
        const int max_steps = MAX_STEPS; // ограничиваем количество шагов за кадр
        int steps = 0;

        while (accumulator >= step && steps < max_steps) {
            if (running) {
                updateSPH();
            }
            accumulator -= step;
            steps++;
        }

        // Если шагов было слишком много и accumulator всё ещё большой,
        // можно просто сбросить его (или уменьшить), чтобы не копить ошибку
        if (accumulator > step * max_steps) {
            accumulator = 0.0; // или clamp
        }

        drawParticles();
    }

private:
    bool mouse_active = false;           // зажата ли левая кнопка мыши
    Vector2 mouse_world_pos;              // позиция курсора в мировых координатах
    float mouse_radius = 50.0f;           // радиус воздействия
    float mouse_strength = 300.0f;         // сила воздействия (подбирается)

    float zoom = 1.0f;           // коэффициент увеличения
    Vector2 offset = {0.0f, 0.0f}; // смещение камеры (панорамирование)
    Vector2 last_mouse_pos;       // для обработки перетаскивания
    bool dragging = false;

    Vector2 screenToWorld(const Vector2i& p) {
        float wx = (p.x() - mSize.x() / 2) / zoom + mSize.x() / 2 - offset.x;
        float wy = (p.y() - mSize.y() / 2) / zoom + mSize.y() / 2 + offset.y;
        wy = BOX_HEIGHT - wy; // инверсия Y (так как в мировых координатах Y растёт вверх)
        return { wx, wy };
    }

    virtual bool scrollEvent(const Vector2i& p, const Vector2f& rel) override {
        // Изменяем zoom (колёсико вверх — приближение, вниз — отдаление)
        float factor = (rel.y() > 0) ? 1.1f : 0.9f;
        zoom *= factor;
        // Ограничиваем диапазон zoom
        const float min_zoom = 0.2f, max_zoom = 5.0f;
        if (zoom < min_zoom) zoom = min_zoom;
        if (zoom > max_zoom) zoom = max_zoom;
        needs_redraw = true;
        return true;
    }


    virtual bool mouseButtonEvent(const Vector2i& p, int button, bool down, int modifiers) override {
        // Левая кнопка — взаимодействие с водой
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            mouse_active = down;
            if (down) {
                mouse_world_pos = screenToWorld(p);
            }
            // Не нужно вызывать updateSPH, просто запросим перерисовку, чтобы показать круг
            // Если отрисовка и так идёт каждый кадр, можно ничего не делать
            return true;
        }
        // Средняя кнопка — панорамирование (dragging)
        else if (button == GLFW_MOUSE_BUTTON_LEFT) {
            dragging = down;
            last_mouse_pos = Vector2{ (float)p.x(), (float)p.y() };
            return true;
        }
        return Screen::mouseButtonEvent(p, button, down, modifiers);
    }

    virtual bool mouseMotionEvent(const Vector2i& p, const Vector2i& rel, int button, int modifiers) override {
        // Если зажата левая — обновляем позицию взаимодействия
        if (mouse_active) {
            mouse_world_pos = screenToWorld(p);
            needs_redraw = true;
            return true;
        }
        // Если зажата средняя — панорамируем сцену
        if (dragging) {
            // Смещение камеры в мировых координатах (с учётом текущего zoom)
            offset.x += (p.x() - last_mouse_pos.x) / zoom;
            offset.y -= (p.y() - last_mouse_pos.y) / zoom; // минус, так как Y экрана направлен вниз
            last_mouse_pos = Vector2{ (float)p.x(), (float)p.y() };
            needs_redraw = true;
            return true;
        }
        return Screen::mouseMotionEvent(p, rel, button, modifiers);
    }

    void drawParticles() {
        NVGcontext* vg = mNVGContext;
        nvgBeginFrame(vg, mSize.x(), mSize.y(), 1.0f);

        // Применяем трансформации камеры (zoom и pan)
        nvgTranslate(vg, mSize.x() / 2, mSize.y() / 2);
        nvgScale(vg, zoom, zoom);
        nvgTranslate(vg, -mSize.x() / 2 + offset.x, -mSize.y() / 2 - offset.y);

        // Фон
        nvgBeginPath(vg);
        nvgRect(vg, 0, 0, BOX_WIDTH, BOX_HEIGHT);
        nvgFillColor(vg, nvgRGBA(20, 20, 30, 255));
        nvgFill(vg);

        // Границы контейнера
        nvgBeginPath(vg);
        nvgRect(vg, 0, 0, BOX_WIDTH, BOX_HEIGHT);
        nvgStrokeColor(vg, nvgRGBA(255, 255, 255, 100));
        nvgStrokeWidth(vg, 2.0f / zoom); // постоянная толщина линии
        nvgStroke(vg);


        //
        // Проверить в отладчике расчет давления 
        //


        // Отрисовка частиц (цвет зависит от давления)
        float particle_radius = 3.0f; // визуальный радиус (spacing/2)
        for (const auto& p : particles) {
            double pressure_ratio = std::min(1.0, std::abs(p.p) / (GAS_CONST * REST_DENS));
            int red = static_cast<int>(255 * pressure_ratio);
            int green = static_cast<int>(255 * (1.0 - pressure_ratio * 0.5));
            int blue = static_cast<int>(255 * (1.0 - pressure_ratio));

            nvgBeginPath(vg);
            nvgCircle(vg, p.pos.x, BOX_HEIGHT - p.pos.y, particle_radius);

            NVGpaint paint = nvgRadialGradient(vg,
                p.pos.x,
                BOX_HEIGHT - p.pos.y,
                0,
                particle_radius * 1.5f,
                nvgRGBA(red, green, blue, 220),
                nvgRGBA(red / 3, green / 3, blue / 3, 50));
            nvgFillPaint(vg, paint);
            nvgFill(vg);
        }

        // Область взаимодействия мыши
        if (mouse_active) {
            nvgBeginPath(vg);
            nvgCircle(vg, mouse_world_pos.x, BOX_HEIGHT - mouse_world_pos.y, mouse_radius);
            nvgStrokeColor(vg, nvgRGBA(255, 255, 255, 128));
            nvgStrokeWidth(vg, 2.0f / zoom);
            nvgStroke(vg);
        }

        // Сбрасываем трансформации для текста (он рисуется в экранных координатах)
        nvgResetTransform(vg);
        nvgFontSize(vg, 14.0f);
        nvgFontFace(vg, "sans");
        nvgFillColor(vg, nvgRGBA(255, 255, 255, 200));
        nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_TOP);
        nvgText(vg, 10, 10, "SPH Fluid Simulation", nullptr);
        nvgText(vg, 10, 30, "Controls: SPACE=Pause, R=Reset, ESC=Exit | Right mouse=repel, Left=drag, Wheel=zoom", nullptr);

        nvgEndFrame(vg);
    }

    void updateSPH() {
        // Build grid
        oneapi::tbb::concurrent_hash_map<std::array<int, 2>, std::vector<int>, CellKeyHashCompare> grid;

        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, particles.size()),
            [&](const oneapi::tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    const auto& pos = particles.at(i).pos;
                    std::array<int, 2> key = {
                        static_cast<int>(std::floor(pos.x / CELL_SIZE)),
                        static_cast<int>(std::floor(pos.y / CELL_SIZE))
                    };
                    oneapi::tbb::concurrent_hash_map<std::array<int, 2>, std::vector<int>, CellKeyHashCompare>::accessor acc;
                    if (grid.insert(acc, key)) {
                        acc->second.reserve(48);
                    }
                    acc->second.push_back(static_cast<int>(i));
                }
            });



//
// Проверить в отладчике расчет давления и сил отталкивания/притяжения для частиц
//

        // Density + Pressure
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, particles.size()),
            [&](const oneapi::tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    auto& p = particles.at(i);
                    std::array<int, 2> key = {
                        static_cast<int>(std::floor(p.pos.x / CELL_SIZE)),
                        static_cast<int>(std::floor(p.pos.y / CELL_SIZE))
                    };

                    double rho = 0.0;

                    for (int dx = -1; dx <= 1; ++dx) {
                        for (int dy = -1; dy <= 1; ++dy) {
                            std::array<int, 2> nkey = { key[0] + dx, key[1] + dy };
                            oneapi::tbb::concurrent_hash_map<std::array<int, 2>, std::vector<int>, CellKeyHashCompare>::const_accessor cacc;
                            if (grid.find(cacc, nkey)) {
                                for (int j : cacc->second) {
                                    double dist_sq = (p.pos.x - particles.at(j).pos.x) * (p.pos.x - particles.at(j).pos.x) +
                                        (p.pos.y - particles.at(j).pos.y) * (p.pos.y - particles.at(j).pos.y);

                                    if (dist_sq < HSQ) {
                                        double h_diff = HSQ - dist_sq;
                                        rho += MASS * POLY6 * h_diff * h_diff * h_diff;
                                    }
                                }
                            }
                        }
                    }

                    p.rho = std::max(rho, REST_DENS * 0.1); // уменьшил множитель с 0.95 до 0.1 для стабилизации плотности
                    
                    // tait EoS
                    const double gamma = 7.0;
                    const double B = GAS_CONST * REST_DENS / gamma;

                    p.p = B * (pow(p.rho / REST_DENS, gamma) - 1.0);
                    //if (p.p < 0.0) p.p = 0.0;        // запрещаем отрицательное давление
                }
            });

        // Forces
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, particles.size()),
            [&](const oneapi::tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    auto& p_i = particles.at(i);
                    std::array<int, 2> key = {
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

                                    const auto& p_j = particles.at(j);
                                    double dist_sq = (p_i.pos - p_j.pos).magnitudeSquared();

                                    if (dist_sq < HSQ && dist_sq > 1e-12) {
                                        double r = std::sqrt(dist_sq);
                                        Vector2 dir = (p_i.pos - p_j.pos) / r;

                                        // Pressure
                                        double pres_kernel = SPIKY_GRAD * (H - r) * (H - r);
                                        double shared_p = MASS * (p_i.p / (p_i.rho * p_i.rho) + p_j.p / (p_j.rho * p_j.rho));
                                        f_press += dir * (-shared_p * pres_kernel);

                                        // Viscosity
                                        Vector2 v_diff = p_j.vel - p_i.vel;
                                        double visc_kernel = VISC_LAP * (H - r);
                                        f_visc += v_diff * (VISC * MASS * visc_kernel / p_j.rho);
                                    }
                                }
                            }
                        }
                    }
                    if (i == 58) {
                        std::cout << "Fp={" << f_press.x << ";" << f_press.y << "}" << " Fv={" << f_visc.x << ";" << f_visc.y << "}" << std::endl;
                    }
                    Vector2 f_grav = { GX * p_i.rho, GY * p_i.rho };
                    p_i.force = f_press + f_visc + f_grav;
                }
            });

        // Integration + Boundaries
        for (auto& p : particles) {
            Vector2 accel = p.force / p.rho;

            if (mouse_active) {
                Vector2 d = p.pos - mouse_world_pos;
                double dist = d.magnitude();
                if (dist < mouse_radius && dist > 1e-6) {
                    double factor = mouse_strength * (1.0 - dist / mouse_radius);
                    Vector2 dir = d.normalized();
                    accel += dir * factor;
                }
            }

            p.vel += accel * DT;
            p.pos += p.vel * DT;

            const double damping = BOUND_DAMPING;   // 0.85

            if (p.pos.x < EPS) { 
                p.vel.x = -p.vel.x * damping; 
                p.pos.x = EPS + 0.001; 
            }
            if (p.pos.x > BOX_WIDTH - EPS) { 
                p.vel.x = -p.vel.x * damping; 
                p.pos.x = BOX_WIDTH - EPS - 0.001; 
            }
            if (p.pos.y < EPS) { 
                p.vel.y = -p.vel.y * damping; 
                p.pos.y = EPS + 0.001; 
            }
            if (p.pos.y > BOX_HEIGHT - EPS) { 
                p.vel.y = -p.vel.y * damping; 
                p.pos.y = BOX_HEIGHT - EPS - 0.001; 
            }
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
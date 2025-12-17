#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <SFML/Graphics.hpp>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/concurrent_hash_map.h>

// Constants
constexpr double H = 16.0;  // Smoothing length
constexpr double HSQ = H * H;
constexpr double DT = 0.0007;  // Time step
constexpr double REST_DENS = 300.0;  // Rest density
constexpr double GAS_CONST = 2000.0;  // Stiffness k
constexpr double VISC = 200.0;  // Viscosity
constexpr double MASS = 2.5;  // Particle mass
constexpr double GX = 0.0, GY = -10.0;  // Gravity
constexpr double EPS = H;  // Boundary epsilon
constexpr double BOUND_DAMPING = -0.5;  // Boundary damping
constexpr double CELL_SIZE = 2 * H;  // Grid cell size
constexpr int BOX_WIDTH = 200;
constexpr int BOX_HEIGHT = 200;
constexpr int NUM_PARTICLES = 400;  // For dam break

// 2D Kernel constants (normalized for 2D)
constexpr double POLY6 = 4.0 / (M_PI * std::pow(H, 8.0));
constexpr double SPIKY_GRAD = -10.0 / (M_PI * std::pow(H, 5.0));  // For gradient scalar
constexpr double VISC_LAP = 40.0 / (M_PI * std::pow(H, 5.0));  // For Laplacian scalar

struct Vector2 {
    double x = 0.0, y = 0.0;

    Vector2& operator+=(const Vector2& other) { x += other.x; y += other.y; return *this; }
    Vector2& operator-=(const Vector2& other) { x -= other.x; y -= other.y; return *this; }
    Vector2& operator*=(double scalar) { x *= scalar; y *= scalar; return *this; }
    Vector2& operator/=(double scalar) { x /= scalar; y /= scalar; return *this; }
    Vector2 operator+(const Vector2& other) const { return { x + other.x, y + other.y }; }
    Vector2 operator-(const Vector2& other) const { return { x - other.x, y - other.y }; }
    Vector2 operator*(double scalar) const { return { x * scalar, y * scalar }; }
    Vector2 operator/(double scalar) const { return { x / scalar, y / scalar }; }
    double magnitude() const { return std::sqrt(x * x + y * y); }
    Vector2 normalize() const { double mag = magnitude(); return mag > 0 ? *this / mag : Vector2{}; }
    double dot(const Vector2& other) const { return x * other.x + y * other.y; }
};

struct Particle {
    Vector2 pos, vel, force;
    double rho = 1.0;  // Density
    double p = 0.0;    // Pressure
};

struct CellKeyHashCompare {
    static size_t hash(const std::array<int, 2>& key) {
        return static_cast<size_t>(key[0]) ^ (static_cast<size_t>(key[1]) << 1);
    }
    static bool equal(const std::array<int, 2>& a, const std::array<int, 2>& b) { return a == b; }
};

int main() {
    // Initialize particles for dam break
    std::vector<Particle> particles(NUM_PARTICLES);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> jitter(-0.5, 0.5);  // Slight jitter for stability
    int idx = 0;
    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 20; ++j) {
            particles[idx++].pos = { 5.0 + i * 2.0 + jitter(gen), 5.0 + j * 2.0 + jitter(gen) };
        }
    }

    // SFML window
    sf::RenderWindow window(sf::VideoMode(BOX_WIDTH, BOX_HEIGHT), "2D SPH Simulation");
    window.setFramerateLimit(60);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();
        }

        // Build grid for neighbors
        oneapi::tbb::concurrent_hash_map<std::array<int, 2>, std::vector<int>, CellKeyHashCompare> grid;
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, particles.size()),
            [&](const oneapi::tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    const auto& pos = particles[i].pos;
                    std::array<int, 2> key = { static_cast<int>(std::floor(pos.x / CELL_SIZE)),
                                              static_cast<int>(std::floor(pos.y / CELL_SIZE)) };
                    typename decltype(grid)::accessor acc;
                    grid.insert(acc, key);
                    acc->second.push_back(static_cast<int>(i));
                }
            });

        // Compute density and pressure (parallel)
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, particles.size()),
            [&](const oneapi::tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    auto& p_i = particles[i];
                    const auto key = std::array<int, 2>{static_cast<int>(std::floor(p_i.pos.x / CELL_SIZE)),
                        static_cast<int>(std::floor(p_i.pos.y / CELL_SIZE))};
                    double rho = 0.0;
                    for (int dx = -1; dx <= 1; ++dx) {
                        for (int dy = -1; dy <= 1; ++dy) {
                            std::array<int, 2> nkey = { key[0] + dx, key[1] + dy };
                            typename decltype(grid)::const_accessor cacc;
                            if (grid.find(cacc, nkey)) {
                                for (int j : cacc->second) {
                                    const auto& p_j = particles[j];
                                    Vector2 rij = p_i.pos - p_j.pos;
                                    double r = rij.magnitude();
                                    if (r < H) {
                                        rho += MASS * POLY6 * std::pow(HSQ - r * r, 3.0);
                                    }
                                }
                            }
                        }
                    }
                    p_i.rho = rho;
                    p_i.p = GAS_CONST * (p_i.rho - REST_DENS);
                }
            });

        // Compute forces (parallel)
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, particles.size()),
            [&](const oneapi::tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    auto& p_i = particles[i];
                    const auto key = std::array<int, 2>{static_cast<int>(std::floor(p_i.pos.x / CELL_SIZE)),
                        static_cast<int>(std::floor(p_i.pos.y / CELL_SIZE))};
                    Vector2 f_press, f_visc;
                    for (int dx = -1; dx <= 1; ++dx) {
                        for (int dy = -1; dy <= 1; ++dy) {
                            std::array<int, 2> nkey = { key[0] + dx, key[1] + dy };
                            typename decltype(grid)::const_accessor cacc;
                            if (grid.find(cacc, nkey)) {
                                for (int j : cacc->second) {
                                    if (i == static_cast<size_t>(j)) continue;
                                    const auto& p_j = particles[j];
                                    Vector2 rij = p_i.pos - p_j.pos;
                                    double r = rij.magnitude();
                                    if (r < H && r > 1e-6) {
                                        Vector2 dir = rij / r;
                                        // Pressure (symmetric)
                                        f_press -= dir * (MASS * (p_i.p / (p_i.rho * p_i.rho) + p_j.p / (p_j.rho * p_j.rho)) * SPIKY_GRAD * std::pow(H - r, 3.0));
                                        // Viscosity
                                        Vector2 v_diff = p_j.vel - p_i.vel;
                                        f_visc += v_diff * (VISC * MASS / (p_j.rho) * VISC_LAP * (H - r));
                                    }
                                }
                            }
                        }
                    }
                    // Gravity (scaled by density)
                    Vector2 f_grav = { GX * p_i.rho, GY * p_i.rho };
                    p_i.force = f_press + f_visc + f_grav;
                }
            });

        // Integrate
        for (auto& p : particles) {
            Vector2 accel = p.force / p.rho;
            p.vel += accel * DT;
            p.pos += p.vel * DT;

            // Boundaries
            if (p.pos.x < EPS) { p.vel.x *= BOUND_DAMPING; p.pos.x = EPS; }
            if (p.pos.x > BOX_WIDTH - EPS) { p.vel.x *= BOUND_DAMPING; p.pos.x = BOX_WIDTH - EPS; }
            if (p.pos.y < EPS) { p.vel.y *= BOUND_DAMPING; p.pos.y = EPS; }
            if (p.pos.y > BOX_HEIGHT - EPS) { p.vel.y *= BOUND_DAMPING; p.pos.y = BOX_HEIGHT - EPS; }
        }

        // Render
        window.clear(sf::Color::Black);
        for (const auto& p : particles) {
            sf::CircleShape circle(2.0f);
            int color_val = std::min(255, static_cast<int>(255 * (p.rho / (2 * REST_DENS))));
            circle.setFillColor(sf::Color(color_val, 0, 255 - color_val));
            circle.setPosition(static_cast<float>(p.pos.x), static_cast<float>(p.pos.y));
            window.draw(circle);
        }
        window.display();
    }

    return 0;
}
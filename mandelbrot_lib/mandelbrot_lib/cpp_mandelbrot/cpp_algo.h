#ifndef MANDELBROT_H
#define MANDELBROT_H

#include <complex>
#include <vector>

class MandelbrotCPP {
public:
    MandelbrotCPP(float escape_radius);

    int compute_escape_iter(float x, float y, int max_iter) const;

    void compute_grid(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, std::vector<int>& results) const;

private:
    float escape_radius;
};

#endif 

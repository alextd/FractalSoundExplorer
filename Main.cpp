#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS
#include "WinAudio.h"
#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#include <iostream>
#include <complex>
#include <math.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <thread>
#include <functional>
#include <mutex>
#include <queue>
#include "gsl/gsl_poly.h"

//Constants
static const int target_fps = 24;
static const int sample_rate = 48000;
static const int max_freq = 4000;
static const int window_w_init = 1280;
static const int window_h_init = 720;
static const int starting_fractal = 0;
static const double escape_radius_sq = 1000.0;
static const char window_name[] = "Fractal Sound Explorer";

//Settings
static int window_w = window_w_init;
static int window_h = window_h_init;
static int orbit_iters = 200;
static double cam_x = 0.0;
static double cam_y = 0.0;
static double cam_zoom = 400.0;
static int cam_x_fp = 0;
static int cam_y_fp = 0;
static double cam_x_dest = cam_x;
static double cam_y_dest = cam_y;
static double cam_zoom_dest = cam_zoom;
static bool sustain = false;
static bool normalized = true;
static bool use_color = false,use_color2 = false;
static bool hide_orbit = true;
static bool hide_label = true;
static bool draw_exponential_orbit = false;
static double jx = 1e8;
static double jy = 1e8;
static int graphics_iters = 1000;
static int frame = 0;
static bool mute = true;
static bool drawIterPoints = false;
static bool freezeOrbit = true, drawFreezeIndex = false;

//Fractal abstraction definition
typedef void (*Fractal)(double&, double&, double, double);
static Fractal fractal = nullptr;

//Blend modes
const sf::BlendMode BlendAlpha(sf::BlendMode::SrcAlpha, sf::BlendMode::OneMinusSrcAlpha, sf::BlendMode::Add,
                               sf::BlendMode::Zero, sf::BlendMode::One, sf::BlendMode::Add);
const sf::BlendMode BlendIgnoreAlpha(sf::BlendMode::One, sf::BlendMode::Zero, sf::BlendMode::Add,
                                     sf::BlendMode::Zero, sf::BlendMode::One, sf::BlendMode::Add);

//Screen utilities
void ScreenToPt(int x, int y, double& px, double& py) {
  px = double(x - window_w / 2) / cam_zoom - cam_x;
  py = double(y - window_h / 2) / cam_zoom - cam_y;
}
void PtToScreen(double px, double py, int& x, int& y) {
  x = int(cam_zoom * (px + cam_x)) + window_w / 2;
  y = int(cam_zoom * (py + cam_y)) + window_h / 2;
}
int PtXToScreen(double px) {
  return int(cam_zoom * (px + cam_x)) + window_w / 2;
}
int PtYToScreen(double py) {
  return int(cam_zoom * (py + cam_y)) + window_h / 2;
}

//All fractal equations
void mandelbrot_hole(double& x, double& y, double cx, double cy) {
  //H-H^2=C
  std::complex<double> one(1, 0);
  std::complex<double> two(2, 0);
  std::complex<double> four(4, 0);
  std::complex<double> C(cx, cy);
  std::complex<double> hole= (one - std::sqrt(one - four * C))/(two);
  x = hole.real();
  y = hole.imag();
  /*
   //(-1 - std::sqrt(1-4*D))/2
  double nx = cx * cx - cy * cy;
  double ny = 2.0 * cx * cy;
  x = cx - nx;
  y = cy - ny;*/
}
void mandelbrot(double& x, double& y, double cx, double cy) {
  double nx = x*x - y*y + cx;
  double ny = 2.0*x*y + cy;
  x = nx;
  y = ny;
}
void complex_square(double& x, double& y) {
  double nx = x * x - y * y;
  double ny = 2.0 * x * y;
  x = nx;
  y = ny;
}
void decardioidify(double& x, double& y, double f) {
  std::complex<double> p(x, y);
  std::complex<double> p2 = f * p;
  p = (1-f) * p - p2 * p2;
  x = p.real();
  y = p.imag();
}
void mandelbrot2(double& x, double& y, double cx, double cy) {
  x += cx; y += cy;
}
void dumb_mandelbrot(double& x, double& y, double cx, double cy) {
  double nx = x * x - y * y;
  double ny = 2.0 * x * y;
  x = nx;
  y = ny;
}
void burning_ship(double& x, double& y, double cx, double cy) {
  double nx = x*x - y*y + cx;
  double ny = 2.0*std::abs(x*y) + cy;
  x = nx;
  y = ny;
}
void feather(double& x, double& y, double cx, double cy) {
  std::complex<double> z(x, y);
  std::complex<double> z2(x*x, y*y);
  std::complex<double> c(cx, cy);
  std::complex<double> one(1.0, 0.0);
  z = z*z*z/(one + z2) + c;
  x = z.real();
  y = z.imag();
}
void sfx(double& x, double& y, double cx, double cy) {
  std::complex<double> z(x, y);
  std::complex<double> c2(cx*cx, cy*cy);
  z = z * (x*x + y*y) - (z * c2);
  x = z.real();
  y = z.imag();
}
void henon(double& x, double& y, double cx, double cy) {
  double nx = 1.0 - cx*x*x + y;
  double ny = cy*x;
  x = nx;
  y = ny;
}
void duffing(double& x, double& y, double cx, double cy) {
  double nx = y;
  double ny = -cy*x + cx*y - y*y*y;
  x = nx;
  y = ny;
}
void ikeda(double& x, double& y, double cx, double cy) {
  double t = 0.4 - 6.0 / (1.0 + x*x + y*y);
  double st = std::sin(t);
  double ct = std::cos(t);
  double nx = 1.0 + cx*(x*ct - y*st);
  double ny = cy*(x*st + y*ct);
  x = nx;
  y = ny;
}
void chirikov(double& x, double& y, double cx, double cy) {
  y += cy*std::sin(x);
  x += cx*y;
}

//List of fractal equations
static const Fractal all_fractals[] = {
  mandelbrot,
  dumb_mandelbrot,
  feather,
  sfx,
  henon,
  duffing,
  ikeda,
  chirikov,
  burning_ship,
};

void PtToPolar(double x, double y, double& theta, double& radius)
{
  if (x == 0 && y == 0)
  {
    radius = 0; theta = 0; return;
  }
  radius = std::sqrt(y * y + x * x);
  theta = -std::atan(y / x);
  if (x < 0) theta += M_PI;
  else if (y > 0) theta += 2*M_PI;
  // * 180 / M_PI
}

void PolarToPt(double theta, double radius, double& x, double& y)
{
  x = radius * std::cos(theta);
  y = - radius * std::sin(theta);
}

//Special Point-Drawing
static enum class SpecialPointType { None, Hole, HoleReflect, CSqMH, Roots, Count } draw_special;
int root_exp = 1;

inline SpecialPointType& operator++(enum SpecialPointType& state, int) {
  const int i = static_cast<int>(state) + 1;
  state = static_cast<SpecialPointType>((i) % static_cast<int>(SpecialPointType::Count));
  return state;
}



void DrawSpecial(double cx, double cy)
{
  int sx, sy;
  double holeX, holeY;
  mandelbrot_hole(holeX, holeY, cx, cy);
  switch (draw_special)
  {
  case SpecialPointType::CSqMH:
  case SpecialPointType::HoleReflect:
  case SpecialPointType::Hole:
    //std::complex<double> iHole(cx, cy);
    //iHole = iHole - (iHole * iHole);
    PtToScreen(holeX, holeY, sx, sy);
    glVertex2i(sx, sy);
    if (draw_special == SpecialPointType::HoleReflect)
    {
      PtToScreen(-holeX, -holeY, sx, sy);
      glVertex2i(sx, sy);
    }

    if(draw_special == SpecialPointType::CSqMH)
    {
      std::complex<double> h(holeX, holeY);
      std::complex<double> c(cx, cy);
      std::complex<double> cSqMH = std::sqrt((c - h) * (c - h));
      PtToScreen(cSqMH.real(), cSqMH.imag(), sx, sy);
      glVertex2i(sx, sy);
    }
    break;

  case SpecialPointType::Roots:
  {
    /* coefficients of P(x) =  -1 + x^5  */
    /*
    double a[6] = { -1, 0, 0, 0, 0, 1 };
    double z[10];

    gsl_poly_complex_workspace* w
      = gsl_poly_complex_workspace_alloc(6);

    gsl_poly_complex_solve(a, 6, w, z);

    gsl_poly_complex_workspace_free(w);

    for (int i = 0; i < 5; i++)
    {
      printf("z%d = %+.18f %+.18f\n",
        i, z[2 * i], z[2 * i + 1]);
    }*/
  }
    break;
  }
}


//Grid-drawing

static enum class GridType { None, Points, Grid, Polar, PolarFocus, Count } draw_grid;
double polarGridTheta, polarGridRadius;

inline GridType& operator++(enum GridType& state, int) {
  const int i = static_cast<int>(state) + 1;
  state = static_cast<GridType>((i) % static_cast<int>(GridType::Count));
  return state;
}

const int grid_detail = 2; //About 2^grid_detail lines

void DrawGrid(const sf::Window& window)
{
  int sx, sy;
  switch(draw_grid)
  {
  case GridType::Grid:
  {
    double minx, miny, maxx, maxy;
    ScreenToPt(0, 0, minx, miny);
    ScreenToPt(window_w, window_h, maxx, maxy);
    double dx = (maxx - minx);
    dx = ldexp(1, ilogb(dx) - grid_detail); //Round down to closest power of 2, and cut into 2^grid_detail chunks

    glLineWidth(1.0f);
    glColor4f(1, 1, 1, 0.5f);
    glBegin(GL_LINES);
    minx = floor(minx / dx) * dx;
    for (double gx = minx; gx < maxx; gx += dx)
    {
      sx = PtXToScreen(gx);
      glVertex2i(sx, 0);    glVertex2i(sx, window_h);
    }
    miny = floor(miny / dx) * dx;
    for (double gy = miny; gy < maxy; gy += dx)
    {
      sy = PtYToScreen(gy);
      glVertex2i(0, sy);    glVertex2i(window_w, sy);
    }

    glEnd();
  }
  [[fallthrough]];
  case GridType::Points:
    glPointSize(12.0f);
    glColor3f(0, 0, 0);
    glBegin(GL_POINTS);
    PtToScreen(0, 0, sx, sy);
    glVertex2i(sx, sy);
    PtToScreen(1, 0, sx, sy);
    glVertex2i(sx, sy);
    PtToScreen(0, -1, sx, sy);
    glVertex2i(sx, sy);
    glEnd();
    glPointSize(10.0f);
    glColor3f(.8f, .8f, 1);
    glBegin(GL_POINTS);
    PtToScreen(0, 0, sx, sy);
    glVertex2i(sx, sy);
    PtToScreen(1, 0, sx, sy);
    glVertex2i(sx, sy);
    PtToScreen(0, -1, sx, sy);
    glVertex2i(sx, sy);
    glEnd();
    break;

  case GridType::Polar:
    for (double radius = .25; radius <= 1.0; radius += .25)
    {
      glLineWidth(1.0f);
      glColor4f(1, 1, 1, 0.5f);
      glBegin(GL_LINE_LOOP);
      for (double theta = 0; theta < 2 * M_PI; theta += M_PI / 30)
      {
        double px, py;
        PolarToPt(theta, radius, px, py);
        PtToScreen(px, py, sx, sy);
        glVertex2i(sx, sy);
      }
      glEnd();
    }
    glBegin(GL_LINES);
    int ox, oy;
    PtToScreen(0, 0, ox, oy);
    for (double theta = 0; theta < 2 * M_PI; theta += M_PI / 4)
    {
      double px, py;
      PolarToPt(theta, 1, px, py);
      PtToScreen(px, py, sx, sy);
      glVertex2i(sx, sy);
      glVertex2i(ox, oy);
    }
    glEnd();
    break;

  case GridType::PolarFocus:
  {
    glLineWidth(1.0f);
    glColor4f(1, 1, 1, 0.5f);
    glBegin(GL_LINE_LOOP);
    double px, py;
    for (double theta = 0; theta < 2 * M_PI; theta += M_PI / 30)
    {
      PolarToPt(theta, polarGridRadius, px, py);
      PtToScreen(px, py, sx, sy);
      glVertex2i(sx, sy);
    }
    glEnd();
    glBegin(GL_LINE_STRIP);
    PtToScreen(polarGridRadius*2, 0, sx, sy);
    glVertex2i(sx, sy);
    PtToScreen(0, 0, sx, sy);
    glVertex2i(sx, sy);
    PolarToPt(polarGridTheta, polarGridRadius*2, px, py);
    PtToScreen(px, py, sx, sy);
    glVertex2i(sx, sy);
    glEnd();
  }
  break;

  }
}

//Color Cycling
//-- code from somehere:
typedef struct {
float r;       // ∈ [0, 1]
float g;       // ∈ [0, 1]
float b;       // ∈ [0, 1]
} rgb;

typedef struct {
  float h;       // ∈ [0, 360]
  float s;       // ∈ [0, 1]
  float v;       // ∈ [0, 1]
} hsv;

rgb hsv2rgb(hsv HSV)
{
  rgb RGB;
  float H = HSV.h, S = HSV.s, V = HSV.v,
    P, Q, T,
    fract;

  (H == 360) ? (H = 0) : (H /= 60);
  fract = H - floor(H);

  P = V * (1 - S);
  Q = V * (1 - S * fract);
  T = V * (1 - S * (1 - fract));

  if (0 <= H && H < 1)
    RGB = { V, T, P };
  else if (1 <= H && H < 2)
    RGB = { Q, V, P };
  else if (2 <= H && H < 3)
    RGB = { P, V, T };
  else if (3 <= H && H < 4)
    RGB = { P, Q, V };
  else if (4 <= H && H < 5)
    RGB = { T, P, V };
  else if (5 <= H && H < 6)
    RGB = { V, P, Q };
  else
    RGB = { 0, 0, 0 };

  return RGB;
}
//-- end code from somewhere
static int color_cycle = 1;
void SetColor(int i)
{
  hsv val{ (i% color_cycle) * 360.f/color_cycle, 1, 1 };
  rgb result = hsv2rgb(val);
  glColor3f(result.r, result.g, result.b);
}

static int draw_cycle = 1;


//drawing a step, gets complicated to draw half-steps
static enum class HalfStep { None, Half, Rot, Count } half_step_mode;

inline HalfStep& operator++(enum HalfStep& state, int) {
  const int i = static_cast<int>(state) + 1;
  state = static_cast<HalfStep>((i) % static_cast<int>(HalfStep::Count));
  return state;
}

void DrawStep(double& x, double& y, double cx, double cy, int i)
{
  int sx, sy;
  double hx = x, hy = y;//copies for half-points

  //Previous point x,y has been plotted in red
  switch (half_step_mode)
  {
  case HalfStep::None:
    //ezpz, draw next point, also in red.
    fractal(x, y, cx, cy);
    PtToScreen(x, y, sx, sy);
    if (color_cycle > 1 && freezeOrbit)
      SetColor(i);
    if( i % draw_cycle == 0)
      glVertex2i(sx, sy);
    break;

  case HalfStep::Half:
    //Apply half-function for half-way point
    complex_square(hx, hy);//TODO: other than mandelbrot
    PtToScreen(hx, hy, sx, sy);

    //Draw line to half-point, blended to yellow
    glColor3f(1.0f, 1.0f, 0.0f);
    glVertex2i(sx, sy);

    //Re-do the point, starting new green line
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex2i(sx, sy);
    
    
    //Apply fractal function for next point
    fractal(x, y, cx, cy);
    PtToScreen(x, y, sx, sy);

    //Blend to yellow to next point
    glColor3f(1.0f, 1.0f, 0.0f);
    glVertex2i(sx, sy);

    //Re-set color to red for the next point
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex2i(sx, sy);
    
    break;
  case HalfStep::Rot: //Todo: other than mandelbrot
    double theta, radius;
    PtToPolar(x, y, theta, radius);
    if (theta > M_PI) theta -= 2* M_PI;
    if (theta < -M_PI) theta += 2* M_PI;


    if (!drawIterPoints)
    {
      for (int i = 0; i < 32; i++)
      {
        float pct = i / 32.0f;
        double t = theta * (1.0 + pct);
        PolarToPt(t, radius, hx, hy);
        PtToScreen(hx, hy, sx, sy);
        glColor3f(1.0f, pct, 0.0f);
        glVertex2i(sx, sy);
      }
    }
    theta *= 2;//TODO not mandelbrot
    PolarToPt(theta, radius, hx, hy);
    PtToScreen(hx, hy, sx, sy);
    //Draw Yellow 1/3 Point
    glColor3f(1.0f, 1.0f, 0.0f);
    glVertex2i(sx, sy);
    
    radius *= radius;
    PolarToPt(theta, radius, hx, hy);
    PtToScreen(hx, hy, sx, sy);
    if (!drawIterPoints) {
      glColor3f(1.0f, 1.0f, 1.0f);
      glVertex2i(sx, sy);
    }


    //Draw Green 1/2 Point
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex2i(sx, sy);


    //Apply fractal function for next point
    fractal(x, y, cx, cy);
    PtToScreen(x, y, sx, sy);

    if (drawIterPoints)
    {
      //Draw Red Point
      glColor3f(1.0f, 0.0f, 0.0f);
      glVertex2i(sx, sy);
    }
    else
    {
      //Blend to yellow to next point
      glColor3f(1.0f, 1.0f, 0.0f);
      glVertex2i(sx, sy);
      glEnd();

      //Draw big points over lines still
      glBegin(GL_POINTS);
      glColor3f(1.0f, 0.0f, 0.0f);
      glVertex2i(sx, sy);
      glEnd();
      glBegin(GL_LINE_STRIP);
    }
    break;
  }
};



//Synthesizer class to inherit Windows Audio.
class Synth : public WinAudio {
public:
  bool audio_reset;
  bool audio_pause;
  double volume;
  double play_x, play_y;
  double play_cx, play_cy;
  double play_nx, play_ny;
  double play_px, play_py;

  Synth(HWND hwnd) : WinAudio(hwnd, sample_rate) {
    audio_reset = true;
    audio_pause = false;
    volume = 8000.0;
    play_x = 0.0;
    play_y = 0.0;
    play_cx = 0.0;
    play_cy = 0.0;
    play_nx = 0.0;
    play_ny = 0.0;
    play_px = 0.0;
    play_py = 0.0;
  }

  void SetPoint(double x, double y) {
    play_nx = x;
    play_ny = y;
    audio_reset = true;
    audio_pause = false;
  }

  virtual bool onGetData(Chunk& data) override {
    //Setup the chunk info
    data.samples = m_samples;
    data.sampleCount = AUDIO_BUFF_SIZE;
    memset(m_samples, 0, sizeof(m_samples));

    //Check if audio needs to reset
    if (audio_reset) {
      m_audio_time = 0;
      play_cx = (jx < 1e8 ? jx : play_nx);
      play_cy = (jy < 1e8 ? jy : play_ny);
      play_x = play_nx;
      play_y = play_ny;
      play_px = play_nx;
      play_py = play_ny;
      mean_x = play_nx;
      mean_y = play_ny;
      volume = 8000.0;
      audio_reset = false;
    }

    //Check if paused
    if (audio_pause) {
      return true;
    }

    //Generate the tones
    const int steps = sample_rate / max_freq;
    for (int i = 0; i < AUDIO_BUFF_SIZE; i+=2) {
      const int j = m_audio_time % steps;
      if (j == 0) {
        play_px = play_x;
        play_py = play_y;
        fractal(play_x, play_y, play_cx, play_cy);
        if (play_x*play_x + play_y*play_y > escape_radius_sq) {
          audio_pause = true;
          return true;
        }

        if (normalized) {
          dpx = play_px - play_cx;
          dpy = play_py - play_cy;
          dx = play_x - play_cx;
          dy = play_y - play_cy;
          if (dx != 0.0 || dy != 0.0) {
            double dpmag = 1.0 / std::sqrt(1e-12 + dpx*dpx + dpy*dpy);
            double dmag = 1.0 / std::sqrt(1e-12 + dx*dx + dy*dy);
            dpx *= dpmag;
            dpy *= dpmag;
            dx *= dmag;
            dy *= dmag;
          }
        } else {
          //Point is relative to mean
          dx = play_x - mean_x;
          dy = play_y - mean_y;
          dpx = play_px - mean_x;
          dpy = play_py - mean_y;
        }

        //Update mean
        mean_x = mean_x*0.99 + play_x*0.01;
        mean_y = mean_y*0.99 + play_y*0.01;

        //Don't let the volume go to infinity, clamp.
        double m = dx*dx + dy*dy;
        if (m > 2.0) {
          dx *= 2.0 / m;
          dy *= 2.0 / m;
        }
        m = dpx*dpx + dpy*dpy;
        if (m > 2.0) {
          dpx *= 2.0 / m;
          dpy *= 2.0 / m;
        }

        //Lose volume over time unless in sustain mode
        if (!sustain) {
          volume *= 0.9992;
        }
      }

      //Cosine interpolation
      double t = double(j) / double(steps);
      t = 0.5 - 0.5*std::cos(t * 3.14159);
      double wx = t*dx + (1.0 - t)*dpx;
      double wy = t*dy + (1.0 - t)*dpy;

      //Save the audio to the 2 channels
      m_samples[i]   = (int16_t)std::min(std::max(wx * volume, -32000.0), 32000.0);
      m_samples[i+1] = (int16_t)std::min(std::max(wy * volume, -32000.0), 32000.0);
      m_audio_time += 1;
    }

    //Return the sound clip
    return !audio_reset;
  }

  int16_t m_samples[AUDIO_BUFF_SIZE];
  int32_t m_audio_time;
  double mean_x;
  double mean_y;
  double dx;
  double dy;
  double dpx;
  double dpy;
};

void PtToString(double x, double y, std::string& str)
{
  // precise stream to print coordinates
  std::ostringstream pointStrStream;
  // Set Fixed -Point Notation
  pointStrStream << std::fixed;
  pointStrStream << std::setprecision(10);
  pointStrStream << x << ", " << -y;
  str = pointStrStream.str();
}

//Change the fractal
void SetFractal(sf::Shader& shader, int type, Synth& synth) {
  shader.setUniform("iType", type);
  jx = jy = 1e8;
  fractal = all_fractals[type];
  normalized = (type == 0);
  synth.audio_pause = true;
  hide_orbit = true;
  frame = 0;
}

//Starting point, C
double px, py, clickx, clicky, cTheta, cRadius, orbit_x = 0, orbit_y = 0;
int orbit_step = 0;
int highlight_index = 0; 
bool findFreezeIndex;
double findx, findy;
double decardioid_amount = 0;
int iStepsToAnti = 1;
std::string cAsString, clickAsString;


static enum class BottomScrollingType{ Brush, Edge,Decard, DecardHalf, Count } bottom_scroll_type;

inline BottomScrollingType& operator++(enum BottomScrollingType& state, int) {
  const int i = static_cast<int>(state) + 1;
  state = static_cast<BottomScrollingType>((i) % static_cast<int>(BottomScrollingType::Count));
  return state;
}

//Start the process at px,py.
void StartPoint(Synth& synth)
{
  PtToString(clickx, clicky, clickAsString);
  px = clickx; py = clicky;
  if (bottom_scroll_type == BottomScrollingType::Decard || bottom_scroll_type == BottomScrollingType::DecardHalf)
    decardioidify(px, py, decardioid_amount);
  hide_orbit = false;
  if (!mute) synth.SetPoint(px, py);
  orbit_x = px;
  orbit_y = py;
  orbit_step = 0;
  PtToPolar(px, py, cTheta, cRadius);
  cTheta *= 180 / M_PI;
  if(jx == 1e8)
    PtToString(px, py, cAsString);
}

//Mouse Point to print
std::string mouseAsString;
double mTheta, mRadius;

//Brush points
std::vector< sf::Vector2<double>>brushPoints;

//Save the Starting Point
double x_save[10] = { 0 }, y_save[10] = { 0 };
double slid_lerp;
bool started_slid;
void SavePoint(int state)
{
  x_save[state] = clickx;
  y_save[state] = clicky;
}
void LoadPoint(int state, Synth& synth)
{
  clickx = x_save[state];
  clicky = y_save[state];
  StartPoint(synth);
}
void SliderPoint(double slider)
{
  started_slid = true;
  slid_lerp = slider;
  clickx = x_save[1] * (1 - slider) + x_save[2] * slider;
  clicky = y_save[1] * (1 - slider) + y_save[2] * slider;
}
void DrawBrushPoints()
{
  glBegin(GL_LINE_STRIP);
  int sx, sy;
  for (auto&& bPoint : brushPoints)
  {
    if (bPoint.x == std::numeric_limits<double>::max())
    {
      glEnd();
      glBegin(GL_LINE_STRIP);
    }
    else
    {
      double bx = bPoint.x, by = bPoint.y;
      if (bottom_scroll_type == BottomScrollingType::Decard || bottom_scroll_type == BottomScrollingType::DecardHalf)
        decardioidify(bx, by, decardioid_amount);
      PtToScreen(bx, by, sx, sy);
      glVertex2i(sx, sy);
    }
  }
  glEnd();
}

//Start point based on click point
double lerp(double a, double b, double f)
{
  return (a * (1.0 - f)) + (b * f);
}
void StartScreenPoint(Synth& synth, int sx, int sy)
{
  started_slid = false;
  //set clickx,clicky
  if (sy < 10 )//Top of screen
    SliderPoint(1.0 * sx / (window_w - 1)); //pixel 0-1979 -> 0-1
  else if (window_h - sy < 10)//BOTTOM of screen
  {
    switch (bottom_scroll_type)
    {
      case BottomScrollingType::Decard:
      {
        decardioid_amount = 1.0 * sx / (window_w - 1);
        frame = 0;

        //doesn't change clickpoint so return here
        return;
      }
      case BottomScrollingType::DecardHalf:
      {
        decardioid_amount = 0.5 * sx / (window_w - 1);
        frame = 0;

        //doesn't change clickpoint so return here
        return;
      }
      case BottomScrollingType::Edge:
      {
        double theta = M_PI * ((sx + 1.0) / (window_w)); //this goes from 1 - window_w instead of 0-window_ so that we get a divisible by 1920 number
        double tx, ty, htx, hty;
        PolarToPt(theta, .5, tx, ty);
        htx = tx; hty = ty;
        complex_square(htx, hty);
        clickx = tx - htx; clicky = ty - hty;
        break;
      }
      case BottomScrollingType::Brush:
      {

        std::vector< sf::Vector2<double>> usablePoints;
        std::copy_if(brushPoints.begin(), brushPoints.end(), std::back_inserter(usablePoints), [](auto bp) {return bp.x != std::numeric_limits<double>::max(); });
        if (usablePoints.empty())  return;//Nothing to do!
        int bCount = static_cast<int>(usablePoints.size());

        double desired = (1.0f* (bCount-1) * sx) / (window_w - 1);
        int index = (int)desired;
        double lerpAmount = desired - index;

        if (index == bCount - 1)
        {
          clickx = usablePoints[index].x;
          clicky = usablePoints[index].y;
        }
        else
        {
          clickx = lerp(usablePoints[index].x, usablePoints[index + 1].x, lerpAmount);
          clicky = lerp(usablePoints[index].y, usablePoints[index + 1].y, lerpAmount);
        }
        break;
      }
    }
  }
  else
    ScreenToPt(sx, sy, clickx, clicky);
  
  //clickpoint is set, use it:
  StartPoint(synth);
}

//Save the Camera View
double cam_x_save[10] = { 0 }, cam_y_save[10] = { 0 }, cam_zoom_save[10] = { 400, 400, 400, 400, 400, 400, 400, 400, 400, 400 };
void SaveZoomState(int state)
{
  cam_x_save[state] = cam_x;
  cam_y_save[state] = cam_y;
  cam_zoom_save[state] = cam_zoom;
}
void LoadZoomState(int state)
{
  cam_x_dest = cam_x = cam_x_save[state];
  cam_y_dest = cam_y = cam_y_save[state];
  cam_zoom_dest = cam_zoom = cam_zoom_save[state];
  frame = 0;
}

// Text input for the point.
bool inputting_x = false, inputting_y = false;
std::string input_str_x = "", input_str_y = "";
void StartInputPoint()
{
  input_str_y = "";
  input_str_x = "";
  inputting_x = true;
  inputting_y = false;
}
void FinalizeInputCoor(sf::Event& event, Synth& synth)
{
  inputting_x = false;
  inputting_y = false;
  double x = input_str_x.empty() ? 0 : std::stod(input_str_x);
  double y = input_str_y.empty() ? 0 : -std::stod(input_str_y);
  if (event.key.shift)
  {
    jx = x; jy = y;
  }
  else
  {
    clickx = x; clicky = y;
  }
  StartPoint(synth);
}
void InputPointKey(const sf::Keyboard::Key key)
{
  if (key == sf::Keyboard::Escape)
  {
    inputting_x = false;
    inputting_y = false;
  }
  else if (key >= sf::Keyboard::Numpad0 && key <= sf::Keyboard::Numpad9)
  {
    if (inputting_x)
      input_str_x += std::to_string(key - sf::Keyboard::Numpad0);
    if (inputting_y)
      input_str_y += std::to_string(key - sf::Keyboard::Numpad0);
  }
  else if (key >= sf::Keyboard::Num0 && key <= sf::Keyboard::Num9)
  {
    if (inputting_x)
      input_str_x += std::to_string(key - sf::Keyboard::Num0);
    if (inputting_y)
      input_str_y += std::to_string(key - sf::Keyboard::Num0);
  }
  else if (key == sf::Keyboard::Period)
  {
    if (inputting_x)
      input_str_x += ".";
    if (inputting_y)
      input_str_y += ".";
  }
  else if (key == sf::Keyboard::Subtract || key == sf::Keyboard::Hyphen)
  {
    if (inputting_x)
      input_str_x += "-";
    if (inputting_y)
      input_str_y += "-";
  }
  else if (key == sf::Keyboard::BackSpace)
  {
    if (inputting_x && !input_str_x.empty())
      input_str_x.pop_back();
    if (inputting_y && !input_str_y.empty())
      input_str_y.pop_back();
  }
  else if (key == sf::Keyboard::Comma)
  {
    inputting_x = false;
    inputting_y = true;
  }
}

//Used whenever the window is created or resized
void resize_window(sf::RenderWindow& window, sf::RenderTexture& rt, const sf::ContextSettings& settings, int w, int h) {
  window_w = w;
  window_h = h;
  rt.create(w, h);
  window.setView(sf::View(sf::FloatRect(0, 0, (float)w, (float)h)));
  frame = 0;
}
void make_window(sf::RenderWindow& window, sf::RenderTexture& rt, const sf::ContextSettings& settings, bool is_fullscreen) {
  window.close();
  sf::VideoMode screenSize;
  if (is_fullscreen) {
    screenSize = sf::VideoMode::getDesktopMode();
    window.create(screenSize, window_name, sf::Style::Fullscreen, settings);
  } else {
    screenSize = sf::VideoMode(window_w_init, window_h_init, 24);
    window.create(screenSize, window_name, sf::Style::Resize | sf::Style::Close, settings);
  }
  resize_window(window, rt, settings, screenSize.width, screenSize.height);
  window.setFramerateLimit(target_fps);
  //window.setVerticalSyncEnabled(true);
  window.setKeyRepeatEnabled(false);
  window.requestFocus();
}

std::string GetTimestamp()
{
  using namespace std::chrono;

  const auto current_time_point{ system_clock::now() };
  const auto current_time{ system_clock::to_time_t(current_time_point) };
  const auto current_localtime{ *std::localtime(&current_time) };
  const auto current_time_since_epoch{ current_time_point.time_since_epoch() };
  const auto current_milliseconds{ duration_cast<milliseconds> (current_time_since_epoch).count() % 1000 };

  std::ostringstream stream;
  stream << std::put_time(&current_localtime, "%m-%d-%y_%H-%M-%S") << "." << std::setw(3) << std::setfill('0') << current_milliseconds;
  return stream.str();
}


//Main entry-point
#if _WIN32
INT WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR lpCmdLine, INT nCmdShow) {
#else
int main(int argc, char *argv[]) {
#endif
  //Capture sf errors
  std::ofstream file("sfml-log.txt");
  std::streambuf* previous = sf::err().rdbuf(file.rdbuf());

  //Make sure shader is supported
  if (!sf::Shader::isAvailable()) {
    std::cerr << "Graphics card does not support shaders" << std::endl;
    return 1;
  }

  //Load the vertex shader
  sf::Shader shader;
  if (!shader.loadFromFile("vert.glsl", sf::Shader::Vertex)) {
    std::cerr << "Failed to compile vertex shader" << std::endl;
    system("pause");
    return 1;
  }

  //Load the fragment shader
  if (!shader.loadFromFile("frag.glsl", sf::Shader::Fragment)) {
      std::cout << "Failed to compile fragment shader" << std::endl;
    system("pause");
    return 1;
  }

  //Load the font
  sf::Font font;
  if (!font.loadFromFile("RobotoMono-Medium.ttf")) {
    std::cerr << "Failed to load font" << std::endl;
    system("pause");
    return 1;
  }

  //Create the full-screen rectangle to draw the shader
  sf::RectangleShape rect;
  rect.setPosition(0, 0);

  //GL settings
  sf::ContextSettings settings;
  settings.depthBits = 24;
  settings.stencilBits = 8;
  settings.antialiasingLevel = 4;
  settings.majorVersion = 3;
  settings.minorVersion = 0;

  //Create the window
  sf::RenderWindow window;
  sf::RenderTexture renderTexture;
  bool is_fullscreen = false;
  bool toggle_fullscreen = false;
  make_window(window, renderTexture, settings, is_fullscreen);

  //Create audio synth
  Synth synth(window.getSystemHandle());

  //Setup the shader
  shader.setUniform("iCam", sf::Vector2f((float)cam_x, (float)cam_y));
  shader.setUniform("iZoom", (float)cam_zoom);
  shader.setUniform("iDecardioid", (float)decardioid_amount);
  shader.setUniform("iStepsToAnti", iStepsToAnti);
  SetFractal(shader, starting_fractal, synth);

  //Start the synth
  synth.play();

  //Main Loop
  bool leftPressed = false;
  bool brushing = false;
  bool dragging = false;
  bool juliaDrag = false;
  bool takeScreenshot = false;
  std::string shot_filename;
  bool showHelpMenu = false;

  sf::Vector2i prevDrag;
  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed) {
        window.close();
        break;
      } else if (event.type == sf::Event::Resized) {
        resize_window(window, renderTexture, settings, event.size.width, event.size.height);
      } else if (event.type == sf::Event::KeyPressed) {
        const sf::Keyboard::Key keycode = event.key.code;
        if (inputting_x || inputting_y)
        {
          if (keycode == sf::Keyboard::Enter)
            FinalizeInputCoor(event, synth);
          else
            InputPointKey(keycode);
        }
        else if (keycode == sf::Keyboard::Escape) {
          window.close();
          break;
        }
        else if (keycode == sf::Keyboard::X)
        {
          StartInputPoint();
        }
        else if (keycode >= sf::Keyboard::F1 && keycode <= sf::Keyboard::F9) {
          SetFractal(shader, keycode - sf::Keyboard::F1, synth);
        }
        else if (keycode >= sf::Keyboard::Num0 && keycode <= sf::Keyboard::Num9) {
          int state = keycode - sf::Keyboard::Num0;
          if (event.key.control)
            SaveZoomState(state);
          else
            LoadZoomState(state);
        }
        else if (keycode >= sf::Keyboard::Numpad0 && keycode <= sf::Keyboard::Numpad9) {
          int state = keycode - sf::Keyboard::Numpad0;
          if (event.key.control)
            SavePoint(state);
          else
            LoadPoint(state, synth);
        } else if (keycode == sf::Keyboard::Tilde) {
          StartPoint(synth);
        } else if (keycode == sf::Keyboard::F11) {
          toggle_fullscreen = true;
        } else if (keycode == sf::Keyboard::D) {
          sustain = !sustain;
        } else if (keycode == sf::Keyboard::G) {
          draw_grid++;
          if (draw_grid == GridType::PolarFocus)
          {
            double px, py;
            const sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);
            ScreenToPt(mouse_pos.x, mouse_pos.y, px, py);
            PtToPolar(px, py, polarGridTheta, polarGridRadius);
          }
        } else if (keycode == sf::Keyboard::C) {
          if (use_color)
          {
            use_color = false;
            use_color2 = true;
          }
          else if (use_color2)
            use_color2 = false;
          else
            use_color = true;
          frame = 0;
        } else if (keycode == sf::Keyboard::R) {
          cam_x = cam_x_dest = 0.0;
          cam_y = cam_y_dest = 0.0;
          cam_zoom = cam_zoom_dest = 400.0;
          frame = 0;
        } else if (keycode == sf::Keyboard::J) {
          if (jx < 1e8) {
            jx = jy = 1e8;
          }
          else
          {
            if (event.key.shift) {
              jx = px; jy = py;
            }
            else {
              juliaDrag = true;
              const sf::Vector2i mousePos = sf::Mouse::getPosition(window);
              ScreenToPt(mousePos.x, mousePos.y, jx, jy);
            }
            PtToString(jx, jy, cAsString);
          }
          synth.audio_pause = true;
          hide_orbit = true;
          frame = 0;
        } else if (keycode == sf::Keyboard::S) {
          takeScreenshot = true;
        } else if (keycode == sf::Keyboard::H) {
          showHelpMenu = !showHelpMenu;
        } else if (keycode == sf::Keyboard::L) {
          hide_label = !hide_label;
        } else if (keycode == sf::Keyboard::V) {
          if (event.key.shift)
          {
            draw_special = SpecialPointType::None;
          }
          else if(draw_special == SpecialPointType::Roots)
          {
            root_exp++;
          }
          else
          {
            draw_special++;
            root_exp = 1;
          }
        } else if (keycode == sf::Keyboard::E) {
          draw_exponential_orbit = !draw_exponential_orbit;
        } else if (keycode == sf::Keyboard::Space) {
          freezeOrbit = !freezeOrbit;
        } else if (keycode == sf::Keyboard::LBracket) {
          if (event.key.alt)
          {
            iStepsToAnti--;
            if (iStepsToAnti < 1) iStepsToAnti = 1;
            frame = 0;
          }
          else if (event.key.shift){
            if (event.key.control)
              orbit_iters -= 1;
            else
              orbit_iters /= 2;
            if (orbit_iters == 0) orbit_iters = 1;
          }
          else {
            graphics_iters /= 2; if (graphics_iters == 0) graphics_iters = 1;
            shader.setUniform("iIters", graphics_iters);
          }
        } else if (keycode == sf::Keyboard::RBracket) {
          if (event.key.alt)
          {
            iStepsToAnti++;
            frame = 0;
          }
          else if (event.key.shift){
            if (event.key.control)
              orbit_iters += 1;
            else
              orbit_iters *= 2;
          }
          else {
            graphics_iters *= 2;
            shader.setUniform("iIters", graphics_iters);
          }
        } else if (keycode == sf::Keyboard::Down) {
          drawFreezeIndex = false;
          highlight_index = 0;
        } else if (keycode == sf::Keyboard::Up) {
          findFreezeIndex = true;

          const sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);
          ScreenToPt(mouse_pos.x, mouse_pos.y, findx, findy);
        } else if (keycode == sf::Keyboard::Left) {
          drawFreezeIndex = true;
          highlight_index -= event.key.control ? 3 : event.key.shift ? 30 : 1; 
          if (highlight_index < 0) highlight_index = 0;
        } else if (keycode == sf::Keyboard::Right) {
          drawFreezeIndex = true;
          highlight_index += event.key.control ? 3 : event.key.shift ? 30 : 1;
          if (highlight_index > orbit_iters) highlight_index = orbit_iters;
        } else if (keycode == sf::Keyboard::F) {
          double x = orbit_x;
          double y = orbit_y;
          int numSteps = event.key.control ? 1 : event.key.shift ? 100 : 10;
          double cx = ((jx < 1e8) ? jx : px);
          double cy = ((jx < 1e8) ? jy : py);
          for (int i = 0; i < numSteps; ++i) {
            fractal(x, y, cx, cy);
            if (x * x + y * y > escape_radius_sq) {
              break;
            }

            orbit_step ++;
          }
          orbit_x = x;
          orbit_y = y;
        } else if (keycode == sf::Keyboard::M) {
          mute = !mute;
          synth.audio_pause = mute;
        } else if (keycode == sf::Keyboard::Z) {
          half_step_mode++;
        } else if (keycode == sf::Keyboard::N) {
          if (event.key.shift)
          {
            color_cycle -= event.key.control ? 10 : 1;
            if (color_cycle < 1) color_cycle = 1;
          }
          else color_cycle += event.key.control ? 10 : 1;
        } else if (keycode == sf::Keyboard::O) {
          if (event.key.shift)
          {
            draw_cycle--;
            if (draw_cycle < 1) draw_cycle = 1;
          }
          else draw_cycle++;
        } else if (keycode == sf::Keyboard::I) {
          bottom_scroll_type++;
          frame = 0;// in case it was decardioid
        } else if (keycode == sf::Keyboard::P) {
          drawIterPoints = !drawIterPoints;
        } else if (keycode == sf::Keyboard::B) {
          brushing = !event.key.shift;
          if (event.key.control)
            if (!brushPoints.empty()) brushPoints.pop_back();

          if(event.key.shift)
            brushPoints.clear();
          else
          {
            double bx, by;
            const sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);
            ScreenToPt(mouse_pos.x, mouse_pos.y, bx, by);
            brushPoints.push_back(sf::Vector2(bx, by));
          }
        }
      } else if (event.type == sf::Event::KeyReleased) {
        if (event.key.code == sf::Keyboard::J) {
          juliaDrag = false;
          frame = 0;
        }
        else if (event.key.code == sf::Keyboard::B) {
          brushing = false;
          brushPoints.push_back(sf::Vector2(std::numeric_limits<double>::max(), 0.0));
        }
      } else if (event.type == sf::Event::MouseWheelMoved) {
        cam_zoom_dest *= std::pow(1.1f, event.mouseWheel.delta);
        const sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);
        cam_x_fp = mouse_pos.x;
        cam_y_fp = mouse_pos.y;
      } else if (event.type == sf::Event::MouseButtonPressed) {
        if (event.mouseButton.button == sf::Mouse::Left) {
          leftPressed = true;
          StartScreenPoint(synth, event.mouseButton.x, event.mouseButton.y);
        } else if (event.mouseButton.button == sf::Mouse::Right) {
          prevDrag = sf::Vector2i(event.mouseButton.x, event.mouseButton.y);
          dragging = true;
        } else if (event.mouseButton.button == sf::Mouse::XButton1) {
          synth.audio_pause = true;
          hide_orbit = true;
        }
      } else if (event.type == sf::Event::MouseButtonReleased) {
        if (event.mouseButton.button == sf::Mouse::Left) {
          leftPressed = false;
        } else if (event.mouseButton.button == sf::Mouse::Right) {
          dragging = false;
        }
      } else if (event.type == sf::Event::MouseMoved) {
        int sx = event.mouseMove.x, sy = event.mouseMove.y;
        double mx, my;
        ScreenToPt(sx, sy, mx, my);

        PtToPolar(mx, my, mTheta, mRadius);
        mTheta *= 180 / M_PI;
        PtToString(mx, my, mouseAsString);

        if (brushing) {
          brushPoints.push_back(sf::Vector2(mx, my));
        }
        if (leftPressed) {
          StartScreenPoint(synth, sx, sy);
        }
        if (dragging) {
          sf::Vector2i curDrag = sf::Vector2i(sx, sy);
          cam_x_dest += (curDrag.x - prevDrag.x) / cam_zoom;
          cam_y_dest += (curDrag.y - prevDrag.y) / cam_zoom;
          prevDrag = curDrag;
          frame = 0;
        }
        if (juliaDrag) {
          jx = mx;
          jy = my;
          frame = 0;
        }
      }
    }

    //Apply zoom
    double fpx, fpy, delta_cam_x, delta_cam_y;
    ScreenToPt(cam_x_fp, cam_y_fp, fpx, fpy);
    cam_zoom = cam_zoom*0.5 + cam_zoom_dest*0.5;
    ScreenToPt(cam_x_fp, cam_y_fp, delta_cam_x, delta_cam_y);
    cam_x_dest += delta_cam_x - fpx;
    cam_y_dest += delta_cam_y - fpy;
    cam_x += delta_cam_x - fpx;
    cam_y += delta_cam_y - fpy;
    cam_x = cam_x*0.8 + cam_x_dest*0.2;
    cam_y = cam_y*0.8 + cam_y_dest*0.2;

    //Create drawing flags for the shader
    const bool hasJulia = (jx < 1e8);
    const bool drawMset = (juliaDrag || !hasJulia);
    const bool drawJset = (juliaDrag || hasJulia);
    const int flags = (drawMset ? 0x01 : 0) | (drawJset ? 0x02 : 0) | (use_color ? 0x04 : 0) | (use_color2 ? 0x10 : 0)
      | (bottom_scroll_type == BottomScrollingType::Decard ? 0x08 : 0)
      | (bottom_scroll_type == BottomScrollingType::DecardHalf ? 0x08 : 0);

    //Set the shader parameters
    const sf::Glsl::Vec2 window_res((float)window_w, (float)window_h);
    shader.setUniform("iResolution", window_res);
    shader.setUniform("iCam", sf::Vector2f((float)cam_x, (float)cam_y));
    shader.setUniform("iZoom", (float)cam_zoom);
    shader.setUniform("iDecardioid", (float)decardioid_amount);
    shader.setUniform("iStepsToAnti", iStepsToAnti);
    shader.setUniform("iFlags", flags);
    shader.setUniform("iJulia", sf::Vector2f((float)jx, (float)jy));
    shader.setUniform("iIters", graphics_iters);
    shader.setUniform("iTime", frame);

    //Draw the full-screen shader to the render texture
    sf::RenderStates states = sf::RenderStates::Default;
    states.blendMode = (frame > 0 ? BlendAlpha : BlendIgnoreAlpha);
    states.shader = &shader;
    rect.setSize(window_res);
    renderTexture.draw(rect, states);
    renderTexture.display();

    //Draw the render texture to the window
    sf::Sprite sprite(renderTexture.getTexture());
    window.clear();
    window.draw(sprite, sf::RenderStates(BlendIgnoreAlpha));

    //Save screen shot if needed
    if (takeScreenshot) {
      window.display();
      const time_t t = std::time(0);
      const tm* now = std::localtime(&t);
      char buffer[128];
      std::strftime(buffer, sizeof(buffer), "pic_%m-%d-%y_%H-%M-%S.png", now);
      const sf::Vector2u windowSize = window.getSize();
      sf::Texture texture;
      texture.create(windowSize.x, windowSize.y);
      texture.update(window);
      texture.copyToImage().saveToFile(buffer);
      takeScreenshot = false;
    }


    int sx, sy;
    //Draw a few grid points for reference
    DrawGrid(window);

    if (inputting_x || inputting_y)
    {
      glPushMatrix();
      sf::RectangleShape dimRect(sf::Vector2f((float)window_w, (float)window_h / 10));
      dimRect.setFillColor(sf::Color(0, 0, 0, 128));
      window.draw(dimRect, sf::RenderStates(BlendAlpha));

      sf::Text input_text;
      input_text.setFont(font);
      input_text.setCharacterSize(24);
      input_text.setFillColor(sf::Color::White);
      input_text.setString(
        "Input: " + input_str_x + ", " + input_str_y);
      input_text.setPosition(20.0f, 5.0f);
      window.draw(input_text);
      glPopMatrix();
    }
    else
    {
      double cx = (hasJulia ? jx : px);
      double cy = (hasJulia ? jy : py);
      double highlightDelta = 0;
      int iStep = 0, iStepsNeeded = 0;

      //Draw the orbit
      if (!hide_orbit) {
        //Draw the lines
        double hx = -50, hy = -50;//highlight point, default out of sight
        double findr = 100;

        glLineWidth(2.0f);
        glPointSize(5.0f);
        glColor3f(1.0f, 0.0f, 0.0f);
        glBegin(drawIterPoints ? GL_POINTS : GL_LINE_STRIP);
        double freeze_x = orbit_x, freeze_y = orbit_y;
        int stepsDone = 0;
        double x = orbit_x;
        double y = orbit_y;
        PtToScreen(x, y, sx, sy);
        glVertex2i(sx, sy);
        if (freezeOrbit && 0 == highlight_index)
        {
          hx = x; hy = y;
        }

        int i, exponential_i = 1;
        double lastX = x , lastY = y, prevX, prevY;
        for (i = 0; i < orbit_iters; ++i) {

          prevX = x, prevY = y;
          if (draw_exponential_orbit && (i+1 != exponential_i))
            fractal(x, y, cx, cy);
          else
          {
            exponential_i <<= 1;
            DrawStep(x, y, cx, cy, i+1);
          }

          iStep++;
          if (iStepsNeeded == 0 && iStep == iStepsToAnti)
          {
            iStep = 0;
            double dx = prevX - x, dy = prevY - y;
#define ANTI_ESCAPE_ACTUALLY 0.00001
#define ANTI_ESCAPE ANTI_ESCAPE_ACTUALLY*ANTI_ESCAPE_ACTUALLY
            if (dx* dx + dy*dy < ANTI_ESCAPE)
            {
              iStepsNeeded = i;
            }
          }

          if (freezeOrbit && i == highlight_index - 1 - iStepsToAnti)
          {
            lastX = x, lastY = y;
          }
          if (freezeOrbit && i == highlight_index - 1)
          {
            highlightDelta = std::sqrt((lastX - x) * (lastX - x) + (lastY - y) * (lastY - y));
            hx = x; hy = y;
          }
          if (findFreezeIndex)
          {
            double newr = (findx - x) * (findx - x) + (findy - y) * (findy - y);
            if (newr < findr)
            {
              findr = newr;
              highlight_index = i + 1;
            }
          }

          if (x * x + y * y > escape_radius_sq) {
            break;
          }
          else if (i < max_freq / target_fps) {
            orbit_x = x;
            orbit_y = y;
            stepsDone = i + 1;
          }
        }
        glEnd();
        findFreezeIndex = false;

        //Draw the starting point
        glPointSize(8.0f);
        glBegin(GL_POINTS);
        PtToScreen(cx, cy, sx, sy);
        glColor3f(1, 1, 1);
        glVertex2i(sx, sy);
        glEnd();
        glPointSize(4.0f);
        glBegin(GL_POINTS);
        glColor3f(0, 1, 1);
        glVertex2i(sx, sy);
        glEnd();

        if (bottom_scroll_type == BottomScrollingType::Decard || bottom_scroll_type == BottomScrollingType::DecardHalf)
        {
          PtToScreen(clickx, clicky, sx, sy);
          glPointSize(8.0f);
          glBegin(GL_POINTS);
          glColor3f(1, 1, 0);
          glVertex2i(sx, sy);
          glEnd();
          glPointSize(4.0f);
          glBegin(GL_POINTS);
          glColor3f(1, 1, 1);
          glVertex2i(sx, sy);
          glEnd();
        }

        //Draw the hole point
        if (draw_special != SpecialPointType::None)
        {
          glPointSize(8.0f);
          glBegin(GL_POINTS);
          glColor3f(1, 0, 0);
          DrawSpecial(cx, cy);
          glEnd();
          glPointSize(4.0f);
          glBegin(GL_POINTS);
          glColor3f(1, 1, 1);
          DrawSpecial(cx, cy);
          glEnd();

          /*
          * I don't really use 'invert hole'
          PtToScreen(iHole.real(), iHole.imag(), sx, sy);
          glPointSize(8.0f);
          glBegin(GL_POINTS);
          glColor3f(0, 1, 0);
          glVertex2i(sx, sy);
          glEnd();
          glPointSize(4.0f);
          glBegin(GL_POINTS);
          glColor3f(1, 1, 1);
          glVertex2i(sx, sy);
          glEnd();
          */
        }

        //Draw highlighted point
        if (drawFreezeIndex)
        {
          glPointSize(10.0f);
          glColor4f(1, 1, 1, .5f);
          glBegin(GL_POINTS);
          PtToScreen(hx, hy, sx, sy);
          glVertex2i(sx, sy);
          if (half_step_mode != HalfStep::None)
          {
            complex_square(hx, hy);//todo other than mandelbrot
            PtToScreen(hx, hy, sx, sy);
            glVertex2i(sx, sy);
          }
          glEnd();
        }

        if (freezeOrbit)
        {
          orbit_x = freeze_x;
          orbit_y = freeze_y;
        }
        else
        {
          orbit_step += stepsDone;
        }
      }

      //Draw point label
      if (!hide_label)
      {
        glPushMatrix();
        sf::Text orbit_stepText;
        orbit_stepText.setFont(font);
        orbit_stepText.setCharacterSize(24);
        orbit_stepText.setFillColor(sf::Color::White);

        sf::RectangleShape dimRect(sf::Vector2f((float)window_w,
          10 + 4 * font.getLineSpacing(24)));
        dimRect.setFillColor(sf::Color(0, 0, 0, 128));
        window.draw(dimRect, sf::RenderStates(BlendAlpha));

        window.pushGLStates();
        orbit_stepText.setPosition(20.0f, 5.0f);
        orbit_stepText.setString(
          "C = " + cAsString + "\n" +
          "Click: " + clickAsString + "\n" +
          "Mouse: " + mouseAsString + "\n" +
          "Step: " + std::to_string(orbit_step) + "\tMax Iterations: " + std::to_string(orbit_iters) + "/" + std::to_string(graphics_iters));
        window.draw(orbit_stepText);
        
        std::string hdstr = "?";
        if (highlightDelta > 0)
        {
          std::ostringstream oss;
          oss << std::fixed << std::setprecision(10) << highlightDelta;
          hdstr = oss.str();
        }

        orbit_stepText.setString(
          "(" + std::to_string(cTheta) + "°, " + std::to_string(cRadius) + ")\n" +
          "(" + std::to_string(mTheta) + "°, " + std::to_string(mRadius) + ")\n" +
          "Highlight: " + std::to_string(highlight_index) + ", Colors: " + std::to_string(color_cycle) + "\n" +
          std::to_string(iStepsNeeded) + " / Dist from " + std::to_string(iStepsToAnti) + " ago: " + hdstr);
        orbit_stepText.setPosition(window_w / 2 + 10.0f, 5.0f);
        window.draw(orbit_stepText);
        window.popGLStates();
        glPopMatrix();
      }
    }


    //Draw brush stroke
    if (!brushPoints.empty())
    {
      glLineWidth(4.0f);
      glColor3f(0,0,0);
      DrawBrushPoints();
      glLineWidth(2.0f);
      glColor3f(1, 1, 1);
      DrawBrushPoints();
    }

    if (started_slid)
    {
      int sx = (int)(slid_lerp * window_w);
      glLineWidth(3.0f);
      glColor3f(0, 0, 0);
      glBegin(GL_LINES);
      glVertex2i(sx, 0);
      glVertex2i(sx, 5);
      glEnd();
      glLineWidth(1.0f);
      glColor3f(1, 1, 1);
      glBegin(GL_LINES);
      glVertex2i(sx, 0);
      glVertex2i(sx, 5);
      glEnd();
    }
    

    //Draw help menu
    if (showHelpMenu) {
      sf::RectangleShape dimRect(sf::Vector2f((float)window_w, (float)window_h));
      dimRect.setFillColor(sf::Color(0,0,0,128));
      window.draw(dimRect, sf::RenderStates(BlendAlpha));
      sf::Text helpMenu;
      helpMenu.setFont(font);
      helpMenu.setCharacterSize(24);
      helpMenu.setFillColor(sf::Color::White);
      helpMenu.setString(
        "  H - Toggle Help Menu                Left Mouse - Click or drag to hear orbits\n"
        "  D - Toggle Audio Dampening         Right Mouse - Drag to pan view\n"
        "  C - Toggle Color                    Side Mouse - Stop orbit and sound\n"
        "F11 - Toggle Fullscreen             Scroll Wheel - Zoom in and out\n"
        "  S - Save Snapshot                            X - Input x,y (then press enter, esc to cancel).\n"
        "  R - Reset View                        Spacebar - Animate/pause current orbit (showing next 200 steps).\n"
        "       # - Load View                  Left/Right - Highlight next point (when paused) ctrl:3x, shift:30x - Down ends\n"
        "  Ctrl-# - Save View                           F - Step current orbit when paused(next 10 steps, ctrl:1, shift:100)\n"
        "  ` - Repeat Point                  Ctrl-Numpad# - Save Point, Numpad# - Load Point\n"  
        "  J - Hold down, move mouse, and      (Drag mouse from top-left corner to top-right corner to blend #1 => #2)   \n"
        "      release to make Julia sets.    \n"
        "      Press again to switch back       V - Draw the hole.\n"
        "  F1 - Mandelbrot Set                  Z - Toggle Showing Mandelbrot half-steps.\n"
        "  F2 - Dumb Mandelbrot                 G - Cycle grid drawing states\n"
        "  F3 - Feather Fractal                 L - Toggle labels\n"
        "  F4 - SFX Fractal                     E - Draw exponential steps (1,2,4,8th step...)\n"
        "  F5 - Hénon Map                     [ ] - Inc/decrease iterations *2 (shift-[] for drawn lines, ctrl-[] steps +1) \n"
        "  F6 - Duffing Map                     Z - Show Mandelbrot Half-steps\n"
        "  F7 - Ikeda Map                       N - Use cyclic colors (each N adds another color, shift-N removes one)\n"
        "  F8 - Chirikov Map                    P - Draw cyclic iterations (each N skips +1 step, shift-N -1)\n"
        "  F9 - Burning Ship                    B - Drag and draw lines. Ctrl-B connects to previous point. Shift-B erases.\n"
        "   I - Toggle bottom-edge slider mode: C=Cardioid Edge, Decardioid amount, C=Brush"
      );
      helpMenu.setPosition(20.0f, 20.0f);
      window.draw(helpMenu);
    }

    //Flip the screen buffer
    window.display();

    //Update shader time if frame blending is needed
    const double xSpeed = std::abs(cam_x - cam_x_dest) * cam_zoom_dest;
    const double ySpeed = std::abs(cam_x - cam_x_dest) * cam_zoom_dest;
    const double zoomSpeed = std::abs(cam_zoom / cam_zoom_dest - 1.0);
    if (xSpeed < 0.2 && ySpeed < 0.2 && zoomSpeed < 0.002) {
      frame += 1;
    } else {
      frame = 1;
    }

    //Toggle full-screen if needed
    if (toggle_fullscreen) {
      toggle_fullscreen = false;
      is_fullscreen = !is_fullscreen;
      make_window(window, renderTexture, settings, is_fullscreen);
    }
  }

  //Stop the synth before quitting
  synth.stop();
  return 0;
}

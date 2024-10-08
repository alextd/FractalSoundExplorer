#version 400 compatibility
#extension GL_ARB_gpu_shader_fp64 : enable
#pragma optionNV(fastmath off)
#pragma optionNV(fastprecision off)

#define FLOAT float
#define VEC2 vec2
#define VEC3 vec3
#define AA_LEVEL 1
#define ESCAPE 1000.0
#define ANTI_ESCAPE_ACTUALLY 0.00001
#define ANTI_ESCAPE ANTI_ESCAPE_ACTUALLY*ANTI_ESCAPE_ACTUALLY
#define PI 3.141592653
#define MAX_STEPS 16

#define FLAG_DRAW_MSET ((iFlags & 0x01) == 0x01)
#define FLAG_DRAW_JSET ((iFlags & 0x02) == 0x02)
#define FLAG_NOREFLECT ((iFlags & 0x04) == 0x04)
#define FLAG_DECARDIOID ((iFlags & 0x08) == 0x08)
#define FLAG_DRAW_IJSET ((iFlags & 0x10) == 0x10)

#define COLOR_GREY (iColorMode == 1)
#define COLOR_INVERSE (iColorMode == 2)
#define COLOR_PERIOD (iColorMode == 3)

uniform vec2 iResolution;
uniform vec2 iCam;
uniform vec2 iJulia;
uniform float iZoom;
uniform float iDecardioid;
uniform float iCPercent;
uniform int iType;
uniform int iIters;
uniform int iFlags;
uniform int iColorMode;
uniform int iTime;
uniform int iStepsToAnti;

#define cx_one VEC2(1.0, 0.0)
VEC2 cx_mul(VEC2 a, VEC2 b) {
  return VEC2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
VEC2 cx_sqr(VEC2 a) {
  FLOAT x2 = a.x*a.x;
  FLOAT y2 = a.y*a.y;
  FLOAT xy = a.x*a.y;
  return VEC2(x2 - y2, xy + xy);
}
VEC2 cx_cube(VEC2 a) {
  FLOAT x2 = a.x*a.x;
  FLOAT y2 = a.y*a.y;
  FLOAT d = x2 - y2;
  return VEC2(a.x*(d - y2 - y2), a.y*(x2 + x2 + d));
}
VEC2 cx_div(VEC2 a, VEC2 b) {
  FLOAT denom = 1.0 / (b.x*b.x + b.y*b.y);
  return VEC2(VEC2(a.x*b.x + a.y*b.y, a.y*b.x - a.x*b.y) * denom);
}
VEC2 cx_sin(VEC2 a) {
  return VEC2(sin(a.x) * cosh(a.y), cos(a.x) * sinh(a.y));
}
VEC2 cx_cos(VEC2 a) {
  return VEC2(cos(a.x) * cosh(a.y), -sin(a.x) * sinh(a.y));
}
VEC2 cx_exp(VEC2 a) {
  return exp(a.x) * VEC2(cos(a.y), sin(a.y));
}

//Fractal equations
VEC2 mandelbrot(VEC2 z, VEC2 c) {
  return cx_sqr(z) + c;
}
VEC2 dumb_mandelbrot(VEC2 z, VEC2 c) {
  return cx_sqr(z);
}
VEC2 burning_ship(VEC2 z, VEC2 c) {
  return VEC2(z.x*z.x - z.y*z.y, 2.0*abs(z.x * z.y)) + c;
}
VEC2 feather(VEC2 z, VEC2 c) {
  return cx_div(cx_cube(z), cx_one + z*z) + c;
}
VEC2 sfx(VEC2 z, VEC2 c) {
  return z * dot(z,z) - cx_mul(z, c*c);
}
VEC2 henon(VEC2 z, VEC2 c) {
  return VEC2(1.0 - c.x*z.x*z.x + z.y, c.y * z.x);
}
VEC2 duffing(VEC2 z, VEC2 c) {
  return VEC2(z.y, -c.y*z.x + c.x*z.y - z.y*z.y*z.y);
}
VEC2 ikeda(VEC2 z, VEC2 c) {
  float t = float(0.4 - 6.0/(1.0 + dot(z,z)));
  FLOAT st = sin(t);
  FLOAT ct = cos(t);
  return VEC2(1.0 + c.x*(z.x*ct - z.y*st), c.y*(z.x*st + z.y*ct));
}
VEC2 chirikov(VEC2 z, VEC2 c) {
  z.y += c.y*sin(z.x);
  z.x += c.x*z.y;
  return z;
}
VEC2 latte(VEC2 z, VEC2 c) {
  return cx_sqr(cx_sqr(z)+cx_one)/(4.0*z*(cx_sqr(z)-cx_one));
}


//For some reason, #define options on what DO_LOOP does with no comments
#if 1
#define DO_LOOP(name) \
  for (i = 0; i < iIters; ++i) { \
    VEC2 ppz = pz; \
    pz = z; \
    z = name(z, c); \
    if (dot(z, z) > ESCAPE) { break; } \
    sumz.x += dot(z - pz, pz - ppz); \
    sumz.y += dot(z - pz, z - pz); \
    sumz.z += dot(z - ppz, z - ppz); \
  }
#elif 1
#define DO_LOOP(name) \
  for (i = 0; i < iIters; ++i) { \
    z = name(z, c); \
    if (dot(z, z) > ESCAPE) { break; } \
  }
#else
#elif 0
#define DO_LOOP(name) \
  for (i = 0; i < iIters; ++i) { \
    if (z.y > 0 ) { faulty = true; break; } \
    z = name(z, c); \
    if (dot(z, z) > ESCAPE) { break; } \
  }
#else
#define DO_LOOP(name) for (i = 0; i < iIters; ++i) { z = name(z, c); }
#endif

#define DO_LOOP_ANTI(name) \
  for (i = 0; i < iIters; ++i) { \
    if(iStep == 0) \
    { \
      pz = z; \
    } \
    iStep++;\
    z = name(z, c); \
    if(iStep == iStepsToAnti) \
    { \
      iStep = 0; \
      VEC2 d = pz - z; \
      if (dot(d,d) < ANTI_ESCAPE) { break; } \
    } \
  }

#define DO_LOOP_NOREFLECT(name) \
  for (i = 0; i < iIters; ++i) { \
    pza[iStep] = z; \
    iStep = (iStep+1) % iStepsToAnti % MAX_STEPS; \
    z = name(z, c); \
    if(i > iStep) \
    { \
      VEC2 d = pza[iStep] - z; \
      VEC2 d2 = -pza[iStep] - z; \
      if (dot(z, z) < 4 && dot(d,d) > dot(d2, d2)) { faulty=true; break; } \
    } \
    if (dot(z, z) > ESCAPE) { break; } \
  }

VEC2 decardioidify(VEC2 p, float f)
{
  return (1-f)*p - cx_sqr(f*p);
}


vec3 fractal(VEC2 z, VEC2 c) {
  bool faulty = false;
  VEC2 pz = z;
  VEC3 sumz = VEC3(0.0, 0.0, 0.0);
  int i, iStep = 0;
  if(FLAG_NOREFLECT)
  {
    VEC2 pza[MAX_STEPS];//store last X numbers
    switch (iType) {
      case 0: DO_LOOP_NOREFLECT(mandelbrot); break;
      case 1: DO_LOOP_NOREFLECT(dumb_mandelbrot); break;
      case 2: DO_LOOP_NOREFLECT(feather); break;
      case 3: DO_LOOP_NOREFLECT(sfx); break;
      case 4: DO_LOOP_NOREFLECT(henon); break;
      case 5: DO_LOOP_NOREFLECT(duffing); break;
      case 6: DO_LOOP_NOREFLECT(ikeda); break;
      case 7: DO_LOOP_NOREFLECT(chirikov); break;
      case 8: DO_LOOP_NOREFLECT(burning_ship); break;
      case 9: DO_LOOP_NOREFLECT(latte); break;
    }
  }
  else if(COLOR_INVERSE)
  {
    VEC2 pz = z;
    switch (iType) {
      case 0: DO_LOOP_ANTI(mandelbrot); break;
      case 1: DO_LOOP_ANTI(dumb_mandelbrot); break;
      case 2: DO_LOOP_ANTI(feather); break;
      case 3: DO_LOOP_ANTI(sfx); break;
      case 4: DO_LOOP_ANTI(henon); break;
      case 5: DO_LOOP_ANTI(duffing); break;
      case 6: DO_LOOP_ANTI(ikeda); break;
      case 7: DO_LOOP_ANTI(chirikov); break;
      case 8: DO_LOOP_ANTI(burning_ship); break;
      case 9: DO_LOOP_ANTI(latte); break;
    }
  }
  else
  {
    switch (iType) {
      case 0: DO_LOOP(mandelbrot); break;
      case 1: DO_LOOP(dumb_mandelbrot); break;
      case 2: DO_LOOP(feather); break;
      case 3: DO_LOOP(sfx); break;
      case 4: DO_LOOP(henon); break;
      case 5: DO_LOOP(duffing); break;
      case 6: DO_LOOP(ikeda); break;
      case 7: DO_LOOP(chirikov); break;
      case 8: DO_LOOP(burning_ship); break;
      case 9: DO_LOOP(latte); break;
    }
  }
  
  if(faulty) {
    // todo: combine with coloration below
    return vec3(0.2,0.0,0.0);
  }
  
    // Iterations ended early:
  if (i != iIters) {

    if(COLOR_GREY)
    {
      //white to grey, for island detection
      float n = float(i)/(iIters-1)*0.5f+0.5f;
      return vec3(n, n, n);
    }

    // Teal coloration:
    float n1 = cos(float(i) * 0.1) * 0.4 + 0.5;
    float n2 = cos(float(i) * 0.1) * 0.3 + 0.6;
    return vec3(0, n1, n2);
  } 

    // Max iterations:
  else
  {
    if (COLOR_PERIOD) {
      // Color by period
      sumz = abs(sumz) / iIters;
      vec3 n1 = sin(abs(sumz * 5.0)) * 0.45 + 0.5;
      return n1;
    } 

    // Black
    return vec3(0.0, 0.0, 0.0);
  }
}

float rand(float s) {
  return fract(sin(s*12.9898) * 43758.5453);
}

void main() {
	//Get normalized screen coordinate
	vec2 screen_pos = gl_FragCoord.xy - (iResolution.xy * 0.5);

  vec3 col = vec3(0.0, 0.0, 0.0);
  for (int i = 0; i < AA_LEVEL; ++i) {
    vec2 dxy = vec2(rand(i*0.54321 + iTime), rand(i*0.12345 + iTime));
    VEC2 c = VEC2((screen_pos + dxy) * vec2(1.0, -1.0) / iZoom - iCam);
    
    
    if(FLAG_DECARDIOID){
      c = decardioidify(c, iDecardioid);
    }

    if (FLAG_DRAW_MSET) {
      if(iCPercent == 1)
        col += fractal(c, c);
      else
        col += fractal(c, VEC2(iCPercent * c.x, iCPercent * c.y));
    }
    if (FLAG_DRAW_JSET) {
      col += fractal(c, iJulia);
    }
    if (FLAG_DRAW_IJSET) {
      col += fractal(iJulia, c);
    }
  }

  col /= AA_LEVEL;
  if (FLAG_DRAW_MSET && (FLAG_DRAW_JSET || FLAG_DRAW_IJSET)) {
    col *= 0.5;
  }
  gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0 / (iTime + 1.0));
}

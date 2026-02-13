// FluidSimCpp - Zero-Copy GPU Fluid Simulation
// Port of the C# FluidSim using OpenGL 4.3 Compute Shaders + SSBOs
// Single API stack: no GPU->CPU->GPU roundtrip needed

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

// --- Constants (matching C# exactly) ---
static constexpr int   MaxParticles     = 10000;
static constexpr int   InitialCount     = 1500;
static constexpr int   GridCellSize     = 10;
static constexpr float GravityY         = 9.81f * 100.0f;   // 981
static constexpr float RepulsionForce   = 2000.0f;
static constexpr float DampingFactor    = 10.0f;
static constexpr float CollisionRadius  = 10.0f;
static constexpr float ParticleMass     = 5.0f;
static constexpr float MouseForce       = -2500.0f;
static constexpr float MouseRadius      = 100.0f;
static constexpr float WallMargin       = 25.0f;
static constexpr float WallForce        = 200.0f + GravityY; // 1181
static constexpr float PhysicsHz        = 66.0f;
static constexpr float PhysicsStep      = 1.0f / PhysicsHz;
static constexpr float Bounce           = -0.2f;
static constexpr float BottomFriction   = 0.5f;
static constexpr int   ParticlesToSpawn = 10;

// --- Globals ---
static int screenWidth  = 800;
static int screenHeight = 450;
static int particleCount = 0;
static int gridCols, gridRows;

static double accumulator = 0.0;

static float mouseX = 0.0f, mouseY = 0.0f;
static bool  isLeftMouseDown  = false;
static bool  isRightMouseDown = false;

// --- OpenGL resources ---
static GLuint ssboPos, ssboVel, ssboAcc, ssboGridHeads, ssboNextParticle;
static GLuint vao;
static GLuint renderProgram;
static GLuint csClearGrid, csBuildGrid, csCalcForces, csUpdateParticles;

// --- Shader compilation helpers ---
static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        fprintf(stderr, "Shader compile error:\n%s\n", log);
    }
    return s;
}

static GLuint linkProgram(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        fprintf(stderr, "Program link error:\n%s\n", log);
    }
    return p;
}

static GLuint createComputeProgram(const char* src) {
    GLuint s = compileShader(GL_COMPUTE_SHADER, src);
    GLuint p = glCreateProgram();
    glAttachShader(p, s);
    glLinkProgram(p);
    glDeleteShader(s);
    GLint ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        fprintf(stderr, "Compute program link error:\n%s\n", log);
    }
    return p;
}

// --- GLSL Compute Shaders ---

static const char* srcClearGrid = R"(
#version 430 core
layout(local_size_x = 32) in;
layout(std430, binding = 3) buffer GridHeads { int gridHeads[]; };
uniform int uTotalCells;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < uTotalCells) gridHeads[idx] = -1;
}
)";

static const char* srcBuildGrid = R"(
#version 430 core
layout(local_size_x = 32) in;
layout(std430, binding = 0) buffer Pos  { vec2 pos[]; };
layout(std430, binding = 3) buffer GridHeads { int gridHeads[]; };
layout(std430, binding = 4) buffer NextPart  { int nextParticle[]; };
uniform int uParticleCount;
uniform int uGridCols;
uniform int uGridRows;
uniform int uGridCellSize;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uParticleCount) return;
    vec2 p = pos[i];
    int cx = int(p.x / uGridCellSize);
    int cy = int(p.y / uGridCellSize);
    cx = clamp(cx, 0, uGridCols - 1);
    cy = clamp(cy, 0, uGridRows - 1);
    int cellIndex = cy * uGridCols + cx;
    int oldHead = atomicExchange(gridHeads[cellIndex], int(i));
    nextParticle[i] = oldHead;
}
)";

static const char* srcCalcForces = R"(
#version 430 core
layout(local_size_x = 32) in;
layout(std430, binding = 0) buffer Pos  { vec2 pos[]; };
layout(std430, binding = 1) buffer Vel  { vec2 vel[]; };
layout(std430, binding = 2) buffer Acc  { vec2 acc[]; };
layout(std430, binding = 3) buffer GridHeads { int gridHeads[]; };
layout(std430, binding = 4) buffer NextPart  { int nextParticle[]; };

uniform int   uParticleCount;
uniform int   uGridCols;
uniform int   uGridRows;
uniform int   uGridCellSize;
uniform int   uWidth;
uniform int   uHeight;
uniform float uWallMargin;
uniform float uWallForce;
uniform float uGravityY;
uniform float uRepulsionForce;
uniform float uDampingFactor;
uniform float uCollisionRadius;
uniform float uParticleMass;
uniform int   uIsMouseLeftDown;
uniform float uMouseX;
uniform float uMouseY;
uniform float uMouseRadius;
uniform float uMouseForce;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uParticleCount) return;

    float forceX = 0.0, forceY = 0.0;
    vec2 myPos = pos[i];
    vec2 myVel = vel[i];
    float myX = myPos.x, myY = myPos.y;

    // Wall forces
    if (myX < uWallMargin) forceX += (uWallMargin - myX) * uWallForce;
    else if (myX > uWidth - uWallMargin) forceX -= (myX - (uWidth - uWallMargin)) * uWallForce;
    if (myY < uWallMargin) forceY += (uWallMargin - myY) * uWallForce;
    else if (myY > uHeight - uWallMargin) forceY -= (myY - (uHeight - uWallMargin)) * uWallForce;

    // Grid neighbor lookup
    int cx = int(myX / uGridCellSize);
    int cy = int(myY / uGridCellSize);
    cx = clamp(cx, 0, uGridCols - 1);
    cy = clamp(cy, 0, uGridRows - 1);

    int startX = max(cx - 1, 0);
    int endX   = min(cx + 1, uGridCols - 1);
    int startY = max(cy - 1, 0);
    int endY   = min(cy + 1, uGridRows - 1);

    float collRadSqr = uCollisionRadius * uCollisionRadius;

    for (int y = startY; y <= endY; y++) {
        int rowOff = y * uGridCols;
        for (int x = startX; x <= endX; x++) {
            int neighborIdx = gridHeads[rowOff + x];
            while (neighborIdx != -1) {
                if (int(i) != neighborIdx) {
                    vec2 nPos = pos[neighborIdx];
                    float offX = myX - nPos.x;
                    float offY = myY - nPos.y;
                    float distSqr = offX * offX + offY * offY;
                    if (distSqr < collRadSqr && distSqr > 0.0001) {
                        float dist = sqrt(distSqr);
                        float factor = (uCollisionRadius - dist) / dist * uRepulsionForce;
                        forceX += offX * factor;
                        forceY += offY * factor;
                        vec2 nVel = vel[neighborIdx];
                        forceX += (nVel.x - myVel.x) * uDampingFactor;
                        forceY += (nVel.y - myVel.y) * uDampingFactor;
                    }
                }
                neighborIdx = nextParticle[neighborIdx];
            }
        }
    }

    float accX = forceX / uParticleMass;
    float accY = uGravityY + forceY / uParticleMass;

    // Mouse attraction
    if (uIsMouseLeftDown == 1) {
        float tmX = uMouseX - myX;
        float tmY = uMouseY - myY;
        float dSq = tmX * tmX + tmY * tmY;
        if (dSq < uMouseRadius * uMouseRadius) {
            float dist = sqrt(dSq);
            float f = uMouseForce / (dist + 1.0);
            accX += tmX * f;
            accY += tmY * f;
        }
    }

    acc[i] = vec2(accX, accY);
}
)";

static const char* srcUpdateParticles = R"(
#version 430 core
layout(local_size_x = 32) in;
layout(std430, binding = 0) buffer Pos { vec2 pos[]; };
layout(std430, binding = 1) buffer Vel { vec2 vel[]; };
layout(std430, binding = 2) buffer Acc { vec2 acc[]; };
uniform int   uParticleCount;
uniform float uDt;
uniform int   uWidth;
uniform int   uHeight;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uParticleCount) return;

    const float bounce = -0.2;
    const float bottomFriction = 0.5;

    vec2 v = vel[i];
    vec2 a = acc[i];
    vec2 p = pos[i];

    v += a * uDt;
    p += v * uDt;

    if (p.x < 0.0) { p.x = 0.0; v.x *= bounce; }
    if (p.y < 0.0) { p.y = 0.0; v.y *= bounce; }
    if (p.x > uWidth)  { p.x = uWidth;  v.x *= bounce; }
    if (p.y > uHeight) { p.y = uHeight; v.y *= bounce; v.x *= bottomFriction; }

    vel[i] = v;
    pos[i] = p;
}
)";

// --- Render Shaders ---
static const char* vertSrc = R"(
#version 430 core
layout(location = 0) in vec2 aPos;
uniform mat4 uProjection;
void main() {
    gl_Position = uProjection * vec4(aPos, 0.0, 1.0);
}
)";

static const char* fragSrc = R"(
#version 430 core
out vec4 FragColor;
void main() {
    FragColor = vec4(0.53, 0.81, 0.92, 1.0); // Sky Blue
}
)";

// --- Helper: set uniform by name ---
static void setUniform1i(GLuint prog, const char* name, int v) {
    glUniform1i(glGetUniformLocation(prog, name), v);
}
static void setUniform1f(GLuint prog, const char* name, float v) {
    glUniform1f(glGetUniformLocation(prog, name), v);
}

// --- Dispatch helper ---
static inline GLuint dispatchSize(int count) {
    return (GLuint)((count + 31) / 32);
}

// --- Init SSBOs ---
static void initSSBOs() {
    auto makeSSBO = [](GLuint& buf, GLuint binding, size_t bytes) {
        glGenBuffers(1, &buf);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)bytes, nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buf);
    };

    makeSSBO(ssboPos,          0, MaxParticles * sizeof(float) * 2);
    makeSSBO(ssboVel,          1, MaxParticles * sizeof(float) * 2);
    makeSSBO(ssboAcc,          2, MaxParticles * sizeof(float) * 2);

    gridCols = (screenWidth / GridCellSize) + 1;
    gridRows = (screenHeight / GridCellSize) + 1;
    int totalCells = gridCols * gridRows;

    makeSSBO(ssboGridHeads,    3, totalCells * sizeof(int));
    makeSSBO(ssboNextParticle, 4, MaxParticles * sizeof(int));
}

// --- Init particles ---
static void initParticles() {
    float initPos[InitialCount * 2];
    float initVel[InitialCount * 2];
    memset(initVel, 0, sizeof(initVel));

    int cx = screenWidth / 2;
    int cy = screenHeight / 2;
    for (int i = 0; i < InitialCount; i++) {
        initPos[i * 2 + 0] = (float)(cx - 100 + rand() % 200);
        initPos[i * 2 + 1] = (float)(cy - 100 + rand() % 200);
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboPos);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(initPos), initPos);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboVel);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(initVel), initVel);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboAcc);
    float zeros[InitialCount * 2] = {};
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(zeros), zeros);

    particleCount = InitialCount;
}

// --- Init render pipeline ---
static void initRenderPipeline() {
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // Bind the position SSBO as vertex attribute source (zero-copy!)
    glBindBuffer(GL_ARRAY_BUFFER, ssboPos);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, nullptr);

    GLuint vs = compileShader(GL_VERTEX_SHADER, vertSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    renderProgram = linkProgram(vs, fs);
}

// --- Spawn particles at mouse ---
static void spawnParticles() {
    if (particleCount >= MaxParticles) return;

    int countBefore = particleCount;
    int actualSpawn = 0;
    float newPos[ParticlesToSpawn * 2];
    float newVel[ParticlesToSpawn * 2];

    for (int k = 0; k < ParticlesToSpawn; k++) {
        if (particleCount >= MaxParticles) break;
        newPos[actualSpawn * 2 + 0] = mouseX + (float)(rand() % 21 - 10);
        newPos[actualSpawn * 2 + 1] = mouseY + (float)(rand() % 21 - 10);
        newVel[actualSpawn * 2 + 0] = 0.0f;
        newVel[actualSpawn * 2 + 1] = 50.0f;
        particleCount++;
        actualSpawn++;
    }

    if (actualSpawn > 0) {
        GLintptr offset = (GLintptr)(countBefore * sizeof(float) * 2);
        GLsizeiptr size = (GLsizeiptr)(actualSpawn * sizeof(float) * 2);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboPos);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, size, newPos);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboVel);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, size, newVel);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboAcc);
        float zeros[ParticlesToSpawn * 2] = {};
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, size, zeros);
    }
}

// --- Physics step ---
static void updateSimulation(float dt) {
    if (isRightMouseDown) spawnParticles();
    if (particleCount == 0) return;

    int totalCells = gridCols * gridRows;

    // 1. Clear grid
    glUseProgram(csClearGrid);
    setUniform1i(csClearGrid, "uTotalCells", totalCells);
    glDispatchCompute(dispatchSize(totalCells), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // 2. Build grid
    glUseProgram(csBuildGrid);
    setUniform1i(csBuildGrid, "uParticleCount", particleCount);
    setUniform1i(csBuildGrid, "uGridCols", gridCols);
    setUniform1i(csBuildGrid, "uGridRows", gridRows);
    setUniform1i(csBuildGrid, "uGridCellSize", GridCellSize);
    glDispatchCompute(dispatchSize(particleCount), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // 3. Calculate forces
    glUseProgram(csCalcForces);
    setUniform1i(csCalcForces, "uParticleCount", particleCount);
    setUniform1i(csCalcForces, "uGridCols", gridCols);
    setUniform1i(csCalcForces, "uGridRows", gridRows);
    setUniform1i(csCalcForces, "uGridCellSize", GridCellSize);
    setUniform1i(csCalcForces, "uWidth", screenWidth);
    setUniform1i(csCalcForces, "uHeight", screenHeight);
    setUniform1f(csCalcForces, "uWallMargin", WallMargin);
    setUniform1f(csCalcForces, "uWallForce", WallForce);
    setUniform1f(csCalcForces, "uGravityY", GravityY);
    setUniform1f(csCalcForces, "uRepulsionForce", RepulsionForce);
    setUniform1f(csCalcForces, "uDampingFactor", DampingFactor);
    setUniform1f(csCalcForces, "uCollisionRadius", CollisionRadius);
    setUniform1f(csCalcForces, "uParticleMass", ParticleMass);
    setUniform1i(csCalcForces, "uIsMouseLeftDown", isLeftMouseDown ? 1 : 0);
    setUniform1f(csCalcForces, "uMouseX", mouseX);
    setUniform1f(csCalcForces, "uMouseY", mouseY);
    setUniform1f(csCalcForces, "uMouseRadius", MouseRadius);
    setUniform1f(csCalcForces, "uMouseForce", MouseForce);
    glDispatchCompute(dispatchSize(particleCount), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // 4. Update particles (Euler integration + boundary collision)
    glUseProgram(csUpdateParticles);
    setUniform1i(csUpdateParticles, "uParticleCount", particleCount);
    setUniform1f(csUpdateParticles, "uDt", dt);
    setUniform1i(csUpdateParticles, "uWidth", screenWidth);
    setUniform1i(csUpdateParticles, "uHeight", screenHeight);
    glDispatchCompute(dispatchSize(particleCount), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// --- GLFW callbacks ---
static void framebufferSizeCallback(GLFWwindow* /*w*/, int width, int height) {
    screenWidth = width;
    screenHeight = height;
    glViewport(0, 0, width, height);
}

// --- Main ---
int main() {
    if (!glfwInit()) {
        fprintf(stderr, "Failed to init GLFW\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight, "Fluid Simulation", nullptr, nullptr);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // VSync off

    int version = gladLoadGL(glfwGetProcAddress);
    if (!version) {
        fprintf(stderr, "Failed to init GLAD\n");
        return 1;
    }

    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    // Init ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 430");

    // Init simulation
    initSSBOs();
    initParticles();

    // Compile compute shaders
    csClearGrid       = createComputeProgram(srcClearGrid);
    csBuildGrid       = createComputeProgram(srcBuildGrid);
    csCalcForces      = createComputeProgram(srcCalcForces);
    csUpdateParticles = createComputeProgram(srcUpdateParticles);

    // Init render pipeline (binds ssboPos as vertex source)
    initRenderPipeline();

    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Input
        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        mouseX = (float)mx;
        mouseY = (float)my;
        isLeftMouseDown  = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)  == GLFW_PRESS;
        isRightMouseDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

        // Delta time with cap
        double now = glfwGetTime();
        float dt = (float)(now - lastTime);
        lastTime = now;
        if (dt > 0.25f) dt = 0.25f;

        // Fixed timestep physics
        accumulator += dt;
        while (accumulator >= PhysicsStep) {
            updateSimulation(PhysicsStep);
            accumulator -= PhysicsStep;
        }

        // --- Render ---
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        if (particleCount > 0) {
            // Memory barrier: ensure compute writes are visible to vertex fetch
            glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

            glUseProgram(renderProgram);

            // Ortho projection (top-left origin, matching C#)
            float L = 0.0f, R = (float)screenWidth;
            float T = 0.0f, B = (float)screenHeight;
            float proj[16] = {
                2.0f/(R-L),    0.0f,          0.0f, 0.0f,
                0.0f,          2.0f/(T-B),    0.0f, 0.0f,
                0.0f,          0.0f,         -1.0f, 0.0f,
                -(R+L)/(R-L), -(T+B)/(T-B),  0.0f, 1.0f,
            };
            glUniformMatrix4fv(glGetUniformLocation(renderProgram, "uProjection"), 1, GL_FALSE, proj);

            glBindVertexArray(vao);
            glDrawArrays(GL_POINTS, 0, particleCount);
        }

        // --- ImGui ---
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Stats window (top-left)
        ImGui::SetNextWindowPos(ImVec2(5, 5), ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.6f);
        ImGui::Begin("Stats", nullptr,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoNav);
        ImGui::TextColored(ImVec4(0, 1, 0, 1), "FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1), "PhysSteps: %.0f Hz", PhysicsHz);
        ImGui::End();

        // Controls window (top-right)
        ImGui::SetNextWindowPos(ImVec2((float)screenWidth - 160, 5), ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.6f);
        ImGui::Begin("Controls", nullptr,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoNav);
        ImGui::Text("Particles: %d", particleCount);
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    glDeleteBuffers(1, &ssboPos);
    glDeleteBuffers(1, &ssboVel);
    glDeleteBuffers(1, &ssboAcc);
    glDeleteBuffers(1, &ssboGridHeads);
    glDeleteBuffers(1, &ssboNextParticle);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(renderProgram);
    glDeleteProgram(csClearGrid);
    glDeleteProgram(csBuildGrid);
    glDeleteProgram(csCalcForces);
    glDeleteProgram(csUpdateParticles);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

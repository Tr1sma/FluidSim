using System;
using System.Numerics;
using Raylib_cs;
using ComputeSharp;

namespace FluidSimulation
{
    public class Program
    {
        public static void Main()
        {
            Raylib.InitWindow(1000, 600, "Fluid Simulation - Accurate Tait SPH + Ghost Walls");
            Raylib.SetTargetFPS(60); 

            using var sim = new Simulation();
            sim.Run();
        }
    }

    public class Simulation : IDisposable
    {
        private const int MaxParticles = 10000; // High resolution
        private int particleCount = 0;
        private const int InitialFluidCount = 4000;

        // SPH Constants
        private const float SmoothingRadius = 20.0f; // Smaller radius for higher resolution
        private const float GridCellSize = SmoothingRadius; 
        private const float RestDensity = 1.0f; 
        
        // Tait EOS Parameters: B = 3500, Gamma = 7
        private const float Stiffness = 5000f; 
        private const float TaitGamma = 7.0f;

        private const float Viscosity = 100f; 
        private const float XSPH_Epsilon = 0.5f; 
        
        private const float ParticleMass = 15.0f; 
        
        // Surface Tension (Becker 2007 / Akinci)
        private const float SurfaceTension = 1500f; 
        private const float SurfaceThreshold = 7.0f; 

        private const float GravityY = 9.81f * 60f; 

        private const float WallDamping = -0.5f;
        
        private const float MouseForce = -1500f;
        private const float MouseRadius = 150f;
        private const int ParticlesToSpawn = 5;

        // GPU Buffers
        private readonly ReadWriteBuffer<Float2> posBuffer;
        private readonly ReadWriteBuffer<Float2> velBuffer;
        private readonly ReadWriteBuffer<Float2> accBuffer;
        private readonly ReadWriteBuffer<float> densityBuffer; 
        private readonly ReadWriteBuffer<int> typeBuffer; 
        
        private readonly ReadWriteBuffer<int> gridHeadsBuffer;
        private readonly ReadWriteBuffer<int> nextParticleBuffer;

        private readonly Float2[] cpuPos;
        private readonly float[] cpuDensity; 
        private readonly int[] cpuType;
        
        private readonly int screenWidth;
        private readonly int screenHeight;
        
        private RenderTexture2D targetTexture;
        private int gridCols;
        private int gridRows;

        private Vector2 mousePosition;
        private MouseButtons currentMouseButtons = MouseButtons.None;

        public Simulation()
        {
            screenWidth = Raylib.GetScreenWidth();
            screenHeight = Raylib.GetScreenHeight();

            gridCols = (int)(screenWidth / GridCellSize) + 2; 
            gridRows = (int)(screenHeight / GridCellSize) + 2;

            GraphicsDevice device = GraphicsDevice.GetDefault();
            
            posBuffer = device.AllocateReadWriteBuffer<Float2>(MaxParticles);
            velBuffer = device.AllocateReadWriteBuffer<Float2>(MaxParticles);
            accBuffer = device.AllocateReadWriteBuffer<Float2>(MaxParticles);
            densityBuffer = device.AllocateReadWriteBuffer<float>(MaxParticles);
            typeBuffer = device.AllocateReadWriteBuffer<int>(MaxParticles);
            
            gridHeadsBuffer = device.AllocateReadWriteBuffer<int>(gridCols * gridRows);
            nextParticleBuffer = device.AllocateReadWriteBuffer<int>(MaxParticles);

            cpuPos = new Float2[MaxParticles];
            cpuDensity = new float[MaxParticles];
            cpuType = new int[MaxParticles];
            
            targetTexture = Raylib.LoadRenderTexture(screenWidth, screenHeight);

            InitializeParticles(screenWidth, screenHeight);
        }

        private void InitializeParticles(int width, int height)
        {
            var rand = new Random();
            Float2[] initPos = new Float2[MaxParticles];
            Float2[] initVel = new Float2[MaxParticles];
            float[] initDensity = new float[MaxParticles];
            int[] initType = new int[MaxParticles];

            int idx = 0;

            // 1. Create Boundary Particles (Type 1) - 3 Layers
            float spacing = SmoothingRadius * 0.5f;
            
            // Bottom Wall
            for (int layer = 0; layer < 3; layer++)
            {
                float y = height - 10 + (layer * spacing); 
                for (float x = 0; x < width; x += spacing)
                {
                    if (idx >= MaxParticles) break;
                    initPos[idx] = new Float2(x, y);
                    initType[idx] = 1;
                    initDensity[idx] = RestDensity;
                    idx++;
                }
            }

            // Left Wall
            for (int layer = 0; layer < 3; layer++)
            {
                float x = 10 - (layer * spacing);
                for (float y = 0; y < height; y += spacing)
                {
                    if (idx >= MaxParticles) break;
                    initPos[idx] = new Float2(x, y);
                    initType[idx] = 1;
                    initDensity[idx] = RestDensity;
                    idx++;
                }
            }

            // Right Wall
            for (int layer = 0; layer < 3; layer++)
            {
                float x = width - 10 + (layer * spacing);
                for (float y = 0; y < height; y += spacing)
                {
                    if (idx >= MaxParticles) break;
                    initPos[idx] = new Float2(x, y);
                    initType[idx] = 1;
                    initDensity[idx] = RestDensity;
                    idx++;
                }
            }
            
            // 2. Create Fluid Particles (Type 0)
            int fluidCols = (int)Math.Sqrt(InitialFluidCount * 2); 
            float fluidSpacing = SmoothingRadius * 0.4f;

            for (int i = 0; i < InitialFluidCount; i++)
            {
                if (idx >= MaxParticles) break;

                float x = (width / 2 - (fluidCols * fluidSpacing)/2) + (i % fluidCols) * fluidSpacing;
                float y = (height / 2 - (fluidCols * fluidSpacing)/2) + (i / fluidCols) * fluidSpacing;
                
                x += (float)(rand.NextDouble() * 2.0 - 1.0);
                y += (float)(rand.NextDouble() * 2.0 - 1.0);

                initPos[idx] = new Float2(x, y);
                initVel[idx] = new Float2(0, 0);
                initDensity[idx] = RestDensity; 
                initType[idx] = 0;
                idx++;
            }

            posBuffer.CopyFrom(initPos, 0, 0, idx);
            velBuffer.CopyFrom(initVel, 0, 0, idx);
            densityBuffer.CopyFrom(initDensity, 0, 0, idx);
            typeBuffer.CopyFrom(initType, 0, 0, idx);

            particleCount = idx;
        }

        public void Run()
        {
            const float PhysicsStep = 0.0006f; 
            double accumulator = 0.0;
            
            while (!Raylib.WindowShouldClose())
            {
                mousePosition = Raylib.GetMousePosition();
                currentMouseButtons = MouseButtons.None;
                if (Raylib.IsMouseButtonDown(MouseButton.Left)) currentMouseButtons |= MouseButtons.Left;
                if (Raylib.IsMouseButtonDown(MouseButton.Right)) currentMouseButtons |= MouseButtons.Right;

                double frameTime = Raylib.GetFrameTime();
                if (frameTime > 0.25) frameTime = 0.25;

                accumulator += frameTime;

                while (accumulator >= PhysicsStep)
                {
                    UpdateSimulation(PhysicsStep);
                    accumulator -= PhysicsStep;
                }

                Render();
            }
            Raylib.CloseWindow();
        }

        private void Render()
        {
            if (particleCount > 0)
            {
                posBuffer.CopyTo(cpuPos, 0, 0, particleCount);
                densityBuffer.CopyTo(cpuDensity, 0, 0, particleCount);
                typeBuffer.CopyTo(cpuType, 0, 0, particleCount);
            }

            Raylib.BeginTextureMode(targetTexture);
            Raylib.ClearBackground(new Color(10, 10, 15, 255));

            for (int i = 0; i < particleCount; i++)
            {
                if (cpuType[i] == 1) // Boundary
                {
                     // Invisible boundaries (or very faint)
                     // Raylib.DrawCircle((int)cpuPos[i].X, (int)cpuPos[i].Y, 2.0f, new Color(30, 30, 30, 255));
                }
                else // Fluid
                {
                    float rho = cpuDensity[i];
                    float n = (rho - RestDensity) / RestDensity; 
                    
                    byte r = (byte)Math.Clamp(n * 255, 0, 255);
                    byte g = (byte)Math.Clamp(100 + n * 155, 100, 255);
                    byte b = 255;
                    
                    Raylib.DrawCircle((int)cpuPos[i].X, (int)cpuPos[i].Y, 3.0f, new Color(r, g, b, (byte)255));
                }
            }

            Raylib.EndTextureMode();

            Raylib.BeginDrawing();
            Raylib.DrawTextureRec(targetTexture.Texture, new Rectangle(0, 0, screenWidth, -screenHeight), Vector2.Zero, Color.White);

            Raylib.DrawFPS(10, 10);
            Raylib.DrawText($"Particles: {particleCount}", 10, 30, 20, Color.White);
            Raylib.EndDrawing();
        }

        private void UpdateSimulation(float deltaTime)
        {
            if ((currentMouseButtons & MouseButtons.Right) != 0) 
            {
                SpawnParticles();
            }

            if (particleCount == 0) return;

            GraphicsDevice device = GraphicsDevice.GetDefault();

            device.For(gridCols * gridRows, new ClearGridShader(gridHeadsBuffer));
            device.For(particleCount, new BuildGridShader(
                gridHeadsBuffer,
                nextParticleBuffer,
                posBuffer,
                gridCols,
                gridRows,
                particleCount,
                (int)GridCellSize
            ));

            device.For(particleCount, new ComputeDensityShader(
                posBuffer,
                gridHeadsBuffer,
                nextParticleBuffer,
                densityBuffer,
                gridCols,
                gridRows,
                particleCount,
                (int)GridCellSize,
                SmoothingRadius,
                ParticleMass
            ));

            device.For(particleCount, new ComputeForcesShader(
                accBuffer,
                posBuffer,
                velBuffer,
                densityBuffer,
                typeBuffer, 
                gridHeadsBuffer,
                nextParticleBuffer,
                gridCols,
                gridRows,
                particleCount,
                (int)GridCellSize,
                SmoothingRadius,
                ParticleMass,
                RestDensity,
                Stiffness,
                Viscosity,
                GravityY,
                screenWidth,
                screenHeight,
                (currentMouseButtons & MouseButtons.Left) != 0 ? 1 : 0,
                mousePosition.X,
                mousePosition.Y,
                MouseRadius,
                MouseForce,
                TaitGamma,
                SurfaceTension,
                SurfaceThreshold,
                XSPH_Epsilon,
                deltaTime
            ));

            device.For(particleCount, new IntegrateShader(
                posBuffer,
                velBuffer,
                accBuffer,
                typeBuffer,
                particleCount,
                deltaTime,
                screenWidth,
                screenHeight,
                WallDamping
            ));
        }

        private void SpawnParticles()
        {
            if (particleCount >= MaxParticles) return;
            var rand = new Random();
            int actualSpawn = 0;
            
            Float2[] newPos = new Float2[ParticlesToSpawn];
            Float2[] newVel = new Float2[ParticlesToSpawn];
            float[] newDensity = new float[ParticlesToSpawn];
            int[] newType = new int[ParticlesToSpawn];

            for (int k = 0; k < ParticlesToSpawn; k++) 
            {
                if (particleCount + actualSpawn >= MaxParticles) break;
                
                newPos[actualSpawn] = new Float2(
                    mousePosition.X + (float)rand.NextDouble() * 10 - 5,
                    mousePosition.Y + (float)rand.NextDouble() * 10 - 5
                );
                newVel[actualSpawn] = new Float2((float)rand.NextDouble() * 20 - 10, 50);
                newDensity[actualSpawn] = RestDensity;
                newType[actualSpawn] = 0; 
                actualSpawn++;
            }

            if (actualSpawn > 0)
            {
                posBuffer.CopyFrom(newPos, 0, particleCount, actualSpawn);
                velBuffer.CopyFrom(newVel, 0, particleCount, actualSpawn);
                densityBuffer.CopyFrom(newDensity, 0, particleCount, actualSpawn);
                typeBuffer.CopyFrom(newType, 0, particleCount, actualSpawn);
                particleCount += actualSpawn;
            }
        }

        public void Dispose()
        {
            posBuffer.Dispose();
            velBuffer.Dispose();
            accBuffer.Dispose();
            densityBuffer.Dispose();
            typeBuffer.Dispose();
            gridHeadsBuffer.Dispose();
            nextParticleBuffer.Dispose();
            Raylib.UnloadRenderTexture(targetTexture);
        }

        [Flags] private enum MouseButtons { None = 0, Left = 1, Right = 2 }
    }

    // --- Shaders ---

    [ComputeSharp.GeneratedComputeShaderDescriptor]
    [ComputeSharp.ThreadGroupSize(64, 1, 1)]
    public readonly partial struct ClearGridShader : IComputeShader
    {
        public readonly ReadWriteBuffer<int> gridHeads;
        public ClearGridShader(ReadWriteBuffer<int> gridHeads) => this.gridHeads = gridHeads;
        public void Execute() => gridHeads[ThreadIds.X] = -1;
    }

    [ComputeSharp.GeneratedComputeShaderDescriptor]
    [ComputeSharp.ThreadGroupSize(64, 1, 1)]
    public readonly partial struct BuildGridShader : IComputeShader
    {
        public readonly ReadWriteBuffer<int> gridHeads;
        public readonly ReadWriteBuffer<int> nextParticle;
        public readonly ReadWriteBuffer<Float2> pos;
        public readonly int gridCols, gridRows, particleCount, gridCellSize;

        public BuildGridShader(ReadWriteBuffer<int> gridHeads, ReadWriteBuffer<int> nextParticle, ReadWriteBuffer<Float2> pos, int gridCols, int gridRows, int particleCount, int gridCellSize)
        {
            this.gridHeads = gridHeads; this.nextParticle = nextParticle; this.pos = pos;
            this.gridCols = gridCols; this.gridRows = gridRows; this.particleCount = particleCount; this.gridCellSize = gridCellSize;
        }

        public void Execute()
        {
            int i = ThreadIds.X;
            if (i >= particleCount) return;

            Float2 p = pos[i];
            int cx = (int)(p.X / gridCellSize);
            int cy = (int)(p.Y / gridCellSize);
            cx = Hlsl.Clamp(cx, 0, gridCols - 1);
            cy = Hlsl.Clamp(cy, 0, gridRows - 1);

            int cellIndex = cy * gridCols + cx;
            int originalHead;
            Hlsl.InterlockedExchange(ref gridHeads[cellIndex], i, out originalHead);
            nextParticle[i] = originalHead;
        }
    }

    [ComputeSharp.GeneratedComputeShaderDescriptor]
    [ComputeSharp.ThreadGroupSize(64, 1, 1)]
    public readonly partial struct ComputeDensityShader : IComputeShader
    {
        public readonly ReadWriteBuffer<Float2> pos;
        public readonly ReadWriteBuffer<int> gridHeads;
        public readonly ReadWriteBuffer<int> nextParticle;
        public readonly ReadWriteBuffer<float> density;
        
        public readonly int gridCols, gridRows, particleCount, gridCellSize;
        public readonly float h; 
        public readonly float mass;

        public ComputeDensityShader(ReadWriteBuffer<Float2> pos, ReadWriteBuffer<int> gridHeads, ReadWriteBuffer<int> nextParticle, ReadWriteBuffer<float> density, int gridCols, int gridRows, int particleCount, int gridCellSize, float h, float mass)
        {
            this.pos = pos; this.gridHeads = gridHeads; this.nextParticle = nextParticle; this.density = density;
            this.gridCols = gridCols; this.gridRows = gridRows; this.particleCount = particleCount; this.gridCellSize = gridCellSize;
            this.h = h; this.mass = mass;
        }

        public void Execute()
        {
            int i = ThreadIds.X;
            if (i >= particleCount) return;

            Float2 p = pos[i];
            float rho = 0f;

            float poly6Const = 4f / (3.14159f * Hlsl.Pow(h, 8));
            float h2 = h * h;

            int cx = (int)(p.X / gridCellSize);
            int cy = (int)(p.Y / gridCellSize);

            for (int y = cy - 1; y <= cy + 1; y++)
            {
                if (y < 0 || y >= gridRows) continue;
                for (int x = cx - 1; x <= cx + 1; x++)
                {
                    if (x < 0 || x >= gridCols) continue;
                    
                    int neighborIdx = gridHeads[y * gridCols + x];
                    while (neighborIdx != -1)
                    {
                        Float2 np = pos[neighborIdx];
                        float dx = p.X - np.X;
                        float dy = p.Y - np.Y;
                        float r2 = dx * dx + dy * dy;

                        if (r2 < h2)
                        {
                            float term = h2 - r2;
                            rho += term * term * term;
                        }
                        neighborIdx = nextParticle[neighborIdx];
                    }
                }
            }

            rho *= poly6Const * mass;
            if (rho < 0.0001f) rho = 0.0001f; 
            density[i] = rho;
        }
    }

    [ComputeSharp.GeneratedComputeShaderDescriptor]
    [ComputeSharp.ThreadGroupSize(64, 1, 1)]
    public readonly partial struct ComputeForcesShader : IComputeShader
    {
        public readonly ReadWriteBuffer<Float2> acc;
        public readonly ReadWriteBuffer<Float2> pos;
        public readonly ReadWriteBuffer<Float2> vel;
        public readonly ReadWriteBuffer<float> density;
        public readonly ReadWriteBuffer<int> type;
        public readonly ReadWriteBuffer<int> gridHeads;
        public readonly ReadWriteBuffer<int> nextParticle;

        public readonly int gridCols, gridRows, particleCount, gridCellSize;
        public readonly float h, mass, restDensity, stiffness, viscosity, gravityY;
        public readonly int screenWidth, screenHeight;
        public readonly int isMouseLeftDown;
        public readonly float mouseX, mouseY, mouseRadius, mouseForce;
        public readonly float taitGamma;
        public readonly float surfaceTension;
        public readonly float surfaceThreshold;
        public readonly float xsphEpsilon;
        public readonly float deltaTime;

        public ComputeForcesShader(ReadWriteBuffer<Float2> acc, ReadWriteBuffer<Float2> pos, ReadWriteBuffer<Float2> vel, ReadWriteBuffer<float> density, ReadWriteBuffer<int> type, ReadWriteBuffer<int> gridHeads, ReadWriteBuffer<int> nextParticle, int gridCols, int gridRows, int particleCount, int gridCellSize, float h, float mass, float restDensity, float stiffness, float viscosity, float gravityY, int screenWidth, int screenHeight, int isMouseLeftDown, float mouseX, float mouseY, float mouseRadius, float mouseForce, float taitGamma, float surfaceTension, float surfaceThreshold, float xsphEpsilon, float deltaTime)
        {
            this.acc = acc; this.pos = pos; this.vel = vel; this.density = density; this.type = type; this.gridHeads = gridHeads; this.nextParticle = nextParticle;
            this.gridCols = gridCols; this.gridRows = gridRows; this.particleCount = particleCount; this.gridCellSize = gridCellSize;
            this.h = h; this.mass = mass; this.restDensity = restDensity; this.stiffness = stiffness; this.viscosity = viscosity; this.gravityY = gravityY;
            this.screenWidth = screenWidth; this.screenHeight = screenHeight;
            this.isMouseLeftDown = isMouseLeftDown; this.mouseX = mouseX; this.mouseY = mouseY; this.mouseRadius = mouseRadius; this.mouseForce = mouseForce;
            this.taitGamma = taitGamma;
            this.surfaceTension = surfaceTension;
            this.surfaceThreshold = surfaceThreshold;
            this.xsphEpsilon = xsphEpsilon;
            this.deltaTime = deltaTime;
        }

        public void Execute()
        {
            int i = ThreadIds.X;
            if (i >= particleCount) return;

            // Don't compute forces FOR wall particles
            if (type[i] == 1) 
            {
                acc[i] = new Float2(0, 0);
                return;
            }

            Float2 p = pos[i];
            Float2 v = vel[i];
            float rho = density[i];
            
            // Tait Equation
            float densityRatio = rho / restDensity;
            if (densityRatio < 0.5f) densityRatio = 0.5f;
            float pressure = stiffness * (Hlsl.Pow(densityRatio, taitGamma) - 1.0f);
            if (pressure < 0) pressure = 0; 

            Float2 force = new Float2(0, 0);
            Float2 xsphVelCorrection = new Float2(0, 0); // XSPH Accumulator

            float spikyGradConst = -30f / (3.14159f * Hlsl.Pow(h, 5));
            float viscLapConst = 40f / (3.14159f * Hlsl.Pow(h, 5));
            float poly6GradConst = -24f / (3.14159f * Hlsl.Pow(h, 8));
            float poly6Const = 4f / (3.14159f * Hlsl.Pow(h, 8)); // For XSPH (W_ij)

            float h2 = h * h;

            int cx = (int)(p.X / gridCellSize);
            int cy = (int)(p.Y / gridCellSize);

            for (int y = cy - 1; y <= cy + 1; y++)
            {
                if (y < 0 || y >= gridRows) continue;
                for (int x = cx - 1; x <= cx + 1; x++)
                {
                    if (x < 0 || x >= gridCols) continue;

                    int neighborIdx = gridHeads[y * gridCols + x];
                    while (neighborIdx != -1)
                    {
                        if (i != neighborIdx)
                        {
                            Float2 np = pos[neighborIdx];
                            float dx = p.X - np.X;
                            float dy = p.Y - np.Y;
                            float r2 = dx * dx + dy * dy;

                            if (r2 < h2 && r2 > 0.00001f)
                            {
                                float r = Hlsl.Sqrt(r2);
                                float h_r = h - r;
                                int nType = type[neighborIdx];

                                if (nType == 1) // BOUNDARY
                                {
                                    float repulsionStrength = 20000.0f; 
                                    if (r < h * 0.8f) 
                                    {
                                        float factor = 1.0f - (r / (h * 0.8f));
                                        float f = repulsionStrength * factor * factor;
                                        force.X += (dx / r) * f;
                                        force.Y += (dy / r) * f;
                                    }
                                    force.X -= v.X * 10.0f; // Stronger friction
                                    force.Y -= v.Y * 10.0f;
                                }
                                else // FLUID
                                {
                                    float neighborRho = density[neighborIdx];
                                    float neighborRatio = neighborRho / restDensity;
                                    if (neighborRatio < 0.5f) neighborRatio = 0.5f;
                                    float neighborPressure = stiffness * (Hlsl.Pow(neighborRatio, taitGamma) - 1.0f);
                                    if (neighborPressure < 0) neighborPressure = 0;

                                    // 1. Pressure
                                    float pTerm = (pressure / (rho * rho)) + (neighborPressure / (neighborRho * neighborRho));
                                    float gradMag = spikyGradConst * h_r * h_r; 
                                    Float2 gradW = new Float2(dx / r * gradMag, dy / r * gradMag);
                                    
                                    force.X -= mass * pTerm * gradW.X;
                                    force.Y -= mass * pTerm * gradW.Y;

                                    // 2. Physical Viscosity
                                    Float2 nv = vel[neighborIdx];
                                    float laplacian = viscLapConst * h_r;
                                    float viscTerm = viscosity * mass / (neighborRho * rho); 
                                    force.X += viscTerm * laplacian * (nv.X - v.X);
                                    force.Y += viscTerm * laplacian * (nv.Y - v.Y);

                                    // 3. Surface Tension
                                    float termSq = (h2 - r2);
                                    float poly6GradMag = poly6GradConst * termSq * termSq * r; 
                                    force.X -= surfaceTension * mass * poly6GradMag * (dx / r);
                                    force.Y -= surfaceTension * mass * poly6GradMag * (dy / r);

                                    // 4. XSPH Artificial Viscosity
                                    // v_corr = epsilon * sum( m/rho * v_rel * W )
                                    float poly6Val = poly6Const * termSq * termSq * termSq;
                                    float xsphFactor = (mass / neighborRho) * poly6Val;
                                    xsphVelCorrection.X += (nv.X - v.X) * xsphFactor;
                                    xsphVelCorrection.Y += (nv.Y - v.Y) * xsphFactor;
                                }
                            }
                        }
                        neighborIdx = nextParticle[neighborIdx];
                    }
                }
            }

            // Convert XSPH Velocity Correction to an Acceleration
            // v_new = v + v_corr -> v_new = v + a * dt
            // a_xsph = v_corr * epsilon / dt
            
            if (deltaTime > 0.000001f)
            {
                force.X += (xsphVelCorrection.X * xsphEpsilon) / deltaTime; // Note: Mass cancels out if we consider this a velocity blend
                force.Y += (xsphVelCorrection.Y * xsphEpsilon) / deltaTime; 
                // Actually, typically XSPH is applied at integration. 
                // But applying as force: F = m * a
                // a = v_corr / dt
                // force += mass * a
                // Let's assume unit mass for the blend or just add to acceleration directly?
                // The accumulator `force` is Force. `acc` is Force/Mass.
                // Let's add it to force, scaled by mass.
                
                // Correction: The formula v_i = v_i + eps * sum(...) is a position update velocity.
                // If we want to effect the velocity state `v`, we treat it as a force.
                // Force XSPH = Mass * (Correction / dt)
                force.X += mass * (xsphVelCorrection.X * xsphEpsilon) / deltaTime;
                force.Y += mass * (xsphVelCorrection.Y * xsphEpsilon) / deltaTime;
            }

            // Mouse Interaction
            if (isMouseLeftDown == 1)
            {
                float dx = mouseX - p.X;
                float dy = mouseY - p.Y;
                float d2 = dx * dx + dy * dy;
                if (d2 < mouseRadius * mouseRadius && d2 > 0.001f)
                {
                    float d = Hlsl.Sqrt(d2);
                    float f = mouseForce * (1.0f - d / mouseRadius);
                    force.X += (dx / d) * f * 50f; 
                    force.Y += (dy / d) * f * 50f;
                }
            }

            // F = ma -> a = F/m
            acc[i] = new Float2(force.X / mass, gravityY + (force.Y / mass));
        }
    }
    [ComputeSharp.GeneratedComputeShaderDescriptor]
    [ComputeSharp.ThreadGroupSize(64, 1, 1)]
    public readonly partial struct IntegrateShader : IComputeShader
    {
        public readonly ReadWriteBuffer<Float2> pos;
        public readonly ReadWriteBuffer<Float2> vel;
        public readonly ReadWriteBuffer<Float2> acc;
        public readonly ReadWriteBuffer<int> type;
        
        public readonly int particleCount;
        public readonly float deltaTime;
        public readonly int width;
        public readonly int height;
        public readonly float wallDamping;

        public IntegrateShader(ReadWriteBuffer<Float2> pos, ReadWriteBuffer<Float2> vel, ReadWriteBuffer<Float2> acc, ReadWriteBuffer<int> type, int particleCount, float deltaTime, int width, int height, float wallDamping)
        {
            this.pos = pos; this.vel = vel; this.acc = acc; this.type = type;
            this.particleCount = particleCount; this.deltaTime = deltaTime; this.width = width; this.height = height; this.wallDamping = wallDamping;
        }
        
        public void Execute()
        {
            int i = ThreadIds.X;
            if (i >= particleCount) return;

            // Freeze boundary particles
            if (type[i] == 1) return;

            Float2 p = pos[i];
            Float2 v = vel[i];
            Float2 a = acc[i];

            // Semi-implicit Euler
            v.X += a.X * deltaTime;
            v.Y += a.Y * deltaTime;
            
            float vSq = v.X * v.X + v.Y * v.Y;
            if (vSq > 4000000.0f) 
            {
                float len = Hlsl.Sqrt(vSq);
                v.X = (v.X / len) * 2000.0f;
                v.Y = (v.Y / len) * 2000.0f;
            }

            p.X += v.X * deltaTime;
            p.Y += v.Y * deltaTime;

            // Simple domain wrap/clamp just in case physics explodes
            if (p.X < 0) p.X = 0;
            if (p.X > width) p.X = width;
            if (p.Y < 0) p.Y = 0;
            if (p.Y > height) p.Y = height;

            vel[i] = v;
            pos[i] = p;
        }
    }
}

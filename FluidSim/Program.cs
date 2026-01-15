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
            Raylib.InitWindow(1000, 600, "Fluid Simulation - SPH 2D");
            Raylib.SetTargetFPS(60); 

            using var sim = new Simulation();
            sim.Run();
        }
    }

    public class Simulation : IDisposable
    {
        private const int MaxParticles = 4000; // Fewer particles but larger
        private int particleCount = 0;
        private const int InitialCount = 2000;

        // SPH Constants
        private const float SmoothingRadius = 24.0f; 
        private const float GridCellSize = SmoothingRadius; 
        private const float RestDensity = 1.0f; 
        private const float Stiffness = 2000f; 
        private const float Viscosity = 300f; 
        
        // Target Spacing ~ 6 pixels. 
        // Mass approx = RestDensity * Spacing^2 = 1 * 36 = 36.
        private const float ParticleMass = 25.0f; 

        private const float GravityY = 9.81f * 60f; 
        private const float WallDamping = -0.5f;
        
        // Interaction
        private const float MouseForce = -1500f;
        private const float MouseRadius = 150f;
        private const int ParticlesToSpawn = 5;

        // GPU Buffers
        private readonly ReadWriteBuffer<Float2> posBuffer;
        private readonly ReadWriteBuffer<Float2> velBuffer;
        private readonly ReadWriteBuffer<Float2> accBuffer;
        private readonly ReadWriteBuffer<float> densityBuffer; 
        
        private readonly ReadWriteBuffer<int> gridHeadsBuffer;
        private readonly ReadWriteBuffer<int> nextParticleBuffer;

        // CPU arrays for rendering
        private readonly Float2[] cpuPos;
        private readonly float[] cpuDensity; 
        
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

            // Setup Grid
            gridCols = (int)(screenWidth / GridCellSize) + 2; 
            gridRows = (int)(screenHeight / GridCellSize) + 2;

            GraphicsDevice device = GraphicsDevice.GetDefault();
            
            posBuffer = device.AllocateReadWriteBuffer<Float2>(MaxParticles);
            velBuffer = device.AllocateReadWriteBuffer<Float2>(MaxParticles);
            accBuffer = device.AllocateReadWriteBuffer<Float2>(MaxParticles);
            densityBuffer = device.AllocateReadWriteBuffer<float>(MaxParticles);
            
            gridHeadsBuffer = device.AllocateReadWriteBuffer<int>(gridCols * gridRows);
            nextParticleBuffer = device.AllocateReadWriteBuffer<int>(MaxParticles);

            cpuPos = new Float2[MaxParticles];
            cpuDensity = new float[MaxParticles];
            
            targetTexture = Raylib.LoadRenderTexture(screenWidth, screenHeight);

            InitializeParticles(screenWidth, screenHeight);
        }

        private void InitializeParticles(int width, int height)
        {
            var rand = new Random();
            Float2[] initPos = new Float2[InitialCount];
            Float2[] initVel = new Float2[InitialCount];
            float[] initDensity = new float[InitialCount];

            int cols = (int)Math.Sqrt(InitialCount * 2); 
            float spacing = SmoothingRadius * 0.4f;

            for (int i = 0; i < InitialCount; i++)
            {
                float x = (width / 2 - (cols * spacing)/2) + (i % cols) * spacing;
                float y = (height / 2 - (cols * spacing)/2) + (i / cols) * spacing;
                
                x += (float)(rand.NextDouble() * 2.0 - 1.0);
                y += (float)(rand.NextDouble() * 2.0 - 1.0);

                initPos[i] = new Float2(x, y);
                initVel[i] = new Float2(0, 0);
                initDensity[i] = RestDensity; 
            }

            posBuffer.CopyFrom(initPos, 0, 0, InitialCount);
            velBuffer.CopyFrom(initVel, 0, 0, InitialCount);
            densityBuffer.CopyFrom(initDensity, 0, 0, InitialCount);

            particleCount = InitialCount;
        }

        public void Run()
        {
            const float PhysicsStep = 0.001f; 
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
            }

            Raylib.BeginTextureMode(targetTexture);
            Raylib.ClearBackground(new Color(15, 15, 25, 255));

            for (int i = 0; i < particleCount; i++)
            {
                float rho = cpuDensity[i];
                float n = (rho - RestDensity) / RestDensity; 
                
                byte r = (byte)Math.Clamp(n * 255, 0, 255);
                byte g = (byte)Math.Clamp(50 + n * 155, 50, 255);
                byte b = 255;
                
                // Draw circle to represent particle volume
                Raylib.DrawCircle((int)cpuPos[i].X, (int)cpuPos[i].Y, 3.0f, new Color(r, g, b, (byte)200));
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
                MouseForce
            ));

            device.For(particleCount, new IntegrateShader(
                posBuffer,
                velBuffer,
                accBuffer,
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

            for (int k = 0; k < ParticlesToSpawn; k++) 
            {
                if (particleCount + actualSpawn >= MaxParticles) break;
                
                newPos[actualSpawn] = new Float2(
                    mousePosition.X + (float)rand.NextDouble() * 10 - 5,
                    mousePosition.Y + (float)rand.NextDouble() * 10 - 5
                );
                newVel[actualSpawn] = new Float2((float)rand.NextDouble() * 20 - 10, 50);
                newDensity[actualSpawn] = RestDensity;
                actualSpawn++;
            }

            if (actualSpawn > 0)
            {
                posBuffer.CopyFrom(newPos, 0, particleCount, actualSpawn);
                velBuffer.CopyFrom(newVel, 0, particleCount, actualSpawn);
                densityBuffer.CopyFrom(newDensity, 0, particleCount, actualSpawn);
                particleCount += actualSpawn;
            }
        }

        public void Dispose()
        {
            posBuffer.Dispose();
            velBuffer.Dispose();
            accBuffer.Dispose();
            densityBuffer.Dispose();
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
            // Clamp to grid
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
        public readonly float h; // Smoothing Radius
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

            // Poly6 Kernel (2D): 4 / (pi * h^8)
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
        public readonly ReadWriteBuffer<int> gridHeads;
        public readonly ReadWriteBuffer<int> nextParticle;

        public readonly int gridCols, gridRows, particleCount, gridCellSize;
        public readonly float h, mass, restDensity, stiffness, viscosity, gravityY;
        public readonly int screenWidth, screenHeight;
        public readonly int isMouseLeftDown;
        public readonly float mouseX, mouseY, mouseRadius, mouseForce;

        public ComputeForcesShader(ReadWriteBuffer<Float2> acc, ReadWriteBuffer<Float2> pos, ReadWriteBuffer<Float2> vel, ReadWriteBuffer<float> density, ReadWriteBuffer<int> gridHeads, ReadWriteBuffer<int> nextParticle, int gridCols, int gridRows, int particleCount, int gridCellSize, float h, float mass, float restDensity, float stiffness, float viscosity, float gravityY, int screenWidth, int screenHeight, int isMouseLeftDown, float mouseX, float mouseY, float mouseRadius, float mouseForce)
        {
            this.acc = acc; this.pos = pos; this.vel = vel; this.density = density; this.gridHeads = gridHeads; this.nextParticle = nextParticle;
            this.gridCols = gridCols; this.gridRows = gridRows; this.particleCount = particleCount; this.gridCellSize = gridCellSize;
            this.h = h; this.mass = mass; this.restDensity = restDensity; this.stiffness = stiffness; this.viscosity = viscosity; this.gravityY = gravityY;
            this.screenWidth = screenWidth; this.screenHeight = screenHeight;
            this.isMouseLeftDown = isMouseLeftDown; this.mouseX = mouseX; this.mouseY = mouseY; this.mouseRadius = mouseRadius; this.mouseForce = mouseForce;
        }

        public void Execute()
        {
            int i = ThreadIds.X;
            if (i >= particleCount) return;

            Float2 p = pos[i];
            Float2 v = vel[i];
            float rho = density[i];
            
            // Pressure calculation with Clamping
            float pressure = stiffness * (rho - restDensity);
            if (pressure < 0) pressure = 0; 

            Float2 force = new Float2(0, 0);

            // Spiky Kernel Gradient Constant (2D): -30 / (pi * h^5)
            float spikyGradConst = -30f / (3.14159f * Hlsl.Pow(h, 5));
            
            // Viscosity Laplacian Constant (2D): 40 / (pi * h^5)
            float viscLapConst = 40f / (3.14159f * Hlsl.Pow(h, 5));
            
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
                                
                                float neighborRho = density[neighborIdx];
                                float neighborPressure = stiffness * (neighborRho - restDensity);
                                if (neighborPressure < 0) neighborPressure = 0;

                                // Pressure Force (Symmetric)
                                float pTerm = (pressure / (rho * rho)) + (neighborPressure / (neighborRho * neighborRho));
                                float gradMag = spikyGradConst * h_r * h_r; // (h-r)^2
                                
                                Float2 gradW = new Float2(dx / r * gradMag, dy / r * gradMag);
                                
                                force.X -= mass * pTerm * gradW.X;
                                force.Y -= mass * pTerm * gradW.Y;

                                // Near-field repulsion (Anti-Clumping)
                                if (r < h * 0.2f)
                                {
                                     float repulsion = 500000.0f * (1.0f - r / (h * 0.2f));
                                     force.X += (dx / r) * repulsion;
                                     force.Y += (dy / r) * repulsion;
                                }

                                // Viscosity Force
                                Float2 nv = vel[neighborIdx];
                                float laplacian = viscLapConst * h_r; // (h-r)
                                
                                float viscTerm = viscosity * mass / (neighborRho * rho); 
                                force.X += viscTerm * laplacian * (nv.X - v.X);
                                force.Y += viscTerm * laplacian * (nv.Y - v.Y);
                            }
                        }
                        neighborIdx = nextParticle[neighborIdx];
                    }
                }
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
                    // Reduced multiplier from 500f to 50f to prevent explosion
                    force.X += (dx / d) * f * 50f; 
                    force.Y += (dy / d) * f * 50f;
                }
            }

            acc[i] = new Float2(force.X, gravityY + force.Y);
        }
    }

    [ComputeSharp.GeneratedComputeShaderDescriptor]
    [ComputeSharp.ThreadGroupSize(64, 1, 1)]
    public readonly partial struct IntegrateShader : IComputeShader
    {
        public readonly ReadWriteBuffer<Float2> pos;
        public readonly ReadWriteBuffer<Float2> vel;
        public readonly ReadWriteBuffer<Float2> acc;
        
        public readonly int particleCount;
        public readonly float deltaTime;
        public readonly int width;
        public readonly int height;
        public readonly float wallDamping;

        public IntegrateShader(ReadWriteBuffer<Float2> pos, ReadWriteBuffer<Float2> vel, ReadWriteBuffer<Float2> acc, int particleCount, float deltaTime, int width, int height, float wallDamping)
        {
            this.pos = pos; this.vel = vel; this.acc = acc;
            this.particleCount = particleCount; this.deltaTime = deltaTime; this.width = width; this.height = height; this.wallDamping = wallDamping;
        }
        
        public void Execute()
        {
            int i = ThreadIds.X;
            if (i >= particleCount) return;

            Float2 p = pos[i];
            Float2 v = vel[i];
            Float2 a = acc[i];

            // Simple Euler Integration with Speed Limit
            v.X += a.X * deltaTime;
            v.Y += a.Y * deltaTime;
            
            // Clamp velocity to prevent explosions (e.g., max 2000 px/s)
            float vSq = v.X * v.X + v.Y * v.Y;
            if (vSq > 4000000.0f) // 2000^2
            {
                float len = Hlsl.Sqrt(vSq);
                v.X = (v.X / len) * 2000.0f;
                v.Y = (v.Y / len) * 2000.0f;
            }

            p.X += v.X * deltaTime;
            p.Y += v.Y * deltaTime;

            // Wall Collisions
            const float margin = 5.0f;
            if (p.X < margin) { p.X = margin; v.X *= wallDamping; }
            if (p.X > width - margin) { p.X = width - margin; v.X *= wallDamping; }
            if (p.Y < margin) { p.Y = margin; v.Y *= wallDamping; }
            if (p.Y > height - margin) { p.Y = height - margin; v.Y *= wallDamping; }

            vel[i] = v;
            pos[i] = p;
        }
    }
}
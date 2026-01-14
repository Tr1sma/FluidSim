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
            Raylib.InitWindow(800, 450, "Fluid Simulation");
            Raylib.SetTargetFPS(9999); 

            using var sim = new Simulation();
            sim.Run();
        }
    }

    public class Simulation : IDisposable
    {
        private const int MaxParticles = 20000; 
        private int particleCount = 0;
        private const int InitialCount = 1500;

        private const float MouseForce = -1000f;
        private const float MouseRadius = 100f;
        private const float WallMargin = 20;
        private const float WallForce = 200f + GravityY;

        private const float GravityY = 9.81f * 100f;

        // GPU Buffers
        private readonly ReadWriteBuffer<float> posXBuffer;
        private readonly ReadWriteBuffer<float> posYBuffer;
        private readonly ReadWriteBuffer<float> velXBuffer;
        private readonly ReadWriteBuffer<float> velYBuffer;
        private readonly ReadWriteBuffer<float> accXBuffer;
        private readonly ReadWriteBuffer<float> accYBuffer;
        
        private readonly ReadWriteBuffer<int> gridHeadsBuffer;
        private readonly ReadWriteBuffer<int> nextParticleBuffer;

        // CPU arrays for initial setup and rendering
        private readonly float[] cpuPosX;
        private readonly float[] cpuPosY;
        // We only need X/Y on CPU for rendering. 
        // We need temp arrays for spawning particles
        
        private const float ParticleMass = 5f;
        private const float CollisionRadius = 10f;
        private const float RepulsionForce = 2000f;
        private const float DampingFactor = 10f;

        private const float PhysikHzRate = 100.0f;

        private const int GridCellSize = 12;
        private int gridCols;
        private int gridRows;

        private Vector2 mousePosition;
        private MouseButtons currentMouseButtons = MouseButtons.None;

        public Simulation()
        {
            int width = Raylib.GetScreenWidth();
            int height = Raylib.GetScreenHeight();

            gridCols = (width / GridCellSize) + 1; 
            gridRows = (height / GridCellSize) + 1;

            // Allocate GPU buffers
            GraphicsDevice device = GraphicsDevice.GetDefault();
            
            posXBuffer = device.AllocateReadWriteBuffer<float>(MaxParticles);
            posYBuffer = device.AllocateReadWriteBuffer<float>(MaxParticles);
            velXBuffer = device.AllocateReadWriteBuffer<float>(MaxParticles);
            velYBuffer = device.AllocateReadWriteBuffer<float>(MaxParticles);
            accXBuffer = device.AllocateReadWriteBuffer<float>(MaxParticles);
            accYBuffer = device.AllocateReadWriteBuffer<float>(MaxParticles);
            
            gridHeadsBuffer = device.AllocateReadWriteBuffer<int>(gridCols * gridRows);
            nextParticleBuffer = device.AllocateReadWriteBuffer<int>(MaxParticles);

            cpuPosX = new float[MaxParticles];
            cpuPosY = new float[MaxParticles];

            InitializeParticles(width, height);
        }

        private void InitializeParticles(int width, int height)
        {
            var rand = new Random();
            float[] initX = new float[InitialCount];
            float[] initY = new float[InitialCount];
            float[] initVelX = new float[InitialCount]; // Zeros
            float[] initVelY = new float[InitialCount]; // Zeros

            for (int i = 0; i < InitialCount; i++)
            {
                initX[i] = rand.Next(width / 2 - 100, width / 2 + 100);
                initY[i] = rand.Next(height / 2 - 100, height / 2 + 100);
            }

            // Upload initial data
            posXBuffer.CopyFrom(initX, 0, 0, InitialCount);
            posYBuffer.CopyFrom(initY, 0, 0, InitialCount);
            // vels and accs are zero already? AllocateReadWriteBuffer does not guarantee zero init.
            // Better to clear them or upload zeros.
            velXBuffer.CopyFrom(initVelX, 0, 0, InitialCount);
            velYBuffer.CopyFrom(initVelY, 0, 0, InitialCount);
            
            // Zero out acceleration just in case
            accXBuffer.CopyFrom(initVelX, 0, 0, InitialCount); // re-using zero array
            accYBuffer.CopyFrom(initVelY, 0, 0, InitialCount);

            particleCount = InitialCount;
        }

        public void Run()
        {
            const float PhysicsStep = 1.0f / PhysikHzRate;
            double accumulator = 0.0;
            GraphicsDevice device = GraphicsDevice.GetDefault();

            // Store frequently used values to avoid constant struct creation if possible,
            // but for ComputeSharp we pass structs.
            
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

                // Retrieve data for rendering
                // We only need the active particles
                if (particleCount > 0)
                {
                    posXBuffer.CopyTo(cpuPosX, 0, 0, particleCount);
                    posYBuffer.CopyTo(cpuPosY, 0, 0, particleCount);
                }

                Raylib.BeginDrawing();
                Raylib.ClearBackground(Color.Black);

                // DrawGridDebug(); 

                for (int i = 0; i < particleCount; i++)
                {
                    // Raylib.DrawCircleV(new Vector2(cpuPosX[i], cpuPosY[i]), 4, Color.SkyBlue);
                    Raylib.DrawPixel((int)cpuPosX[i], (int)cpuPosY[i], Color.SkyBlue);
                }

                Raylib.DrawRectangle(5, 5, 100, 25, new Color(0, 0, 0, 160));
                Raylib.DrawText($"FPS: {Raylib.GetFPS()}", 5, 5, 20, Color.Lime);
                Raylib.DrawText($"PhysSteps: {1.0f / PhysicsStep:F0} Hz", 5, 30, 10, Color.Gray);

                string countText = $"Particles: {particleCount}";
                Raylib.DrawRectangle(Raylib.GetScreenWidth() - 160, 5, 150, 45, new Color(0, 0, 0, 160));
                Raylib.DrawText(countText, Raylib.GetScreenWidth() - 155, 10, 20, Color.White);

                Raylib.EndDrawing();
            }
            Raylib.CloseWindow();
        }

        private void UpdateSimulation(float deltaTime)
        {
            // Spawn particles on CPU and upload
            if ((currentMouseButtons & MouseButtons.Right) != 0) 
            {
                SpawnParticles();
            }

            if (particleCount == 0) return;

            GraphicsDevice device = GraphicsDevice.GetDefault();

            // 1. Clear Grid
            int totalGridCells = gridCols * gridRows;
            device.For(totalGridCells, new ClearGridShaderV2(gridHeadsBuffer));

            // 2. Build Grid
            device.For(particleCount, new BuildGridShaderV2(
                gridHeadsBuffer,
                nextParticleBuffer,
                posXBuffer,
                posYBuffer,
                gridCols,
                gridRows,
                particleCount,
                GridCellSize
            ));

            // 3. Calculate Forces
            device.For(particleCount, new CalculateForcesShaderV2(
                accXBuffer,
                accYBuffer,
                posXBuffer,
                posYBuffer,
                velXBuffer,
                velYBuffer,
                gridHeadsBuffer,
                nextParticleBuffer,
                gridCols,
                gridRows,
                particleCount,
                GridCellSize,
                Raylib.GetScreenWidth(),
                Raylib.GetScreenHeight(),
                WallMargin,
                WallForce,
                GravityY,
                RepulsionForce,
                DampingFactor,
                CollisionRadius,
                ParticleMass,
                (currentMouseButtons & MouseButtons.Left) != 0 ? 1 : 0,
                mousePosition.X,
                mousePosition.Y,
                MouseRadius,
                MouseForce
            ));

            // 4. Update Particles
            device.For(particleCount, new UpdateParticlesShaderV2(
                posXBuffer,
                posYBuffer,
                velXBuffer,
                velYBuffer,
                accXBuffer,
                accYBuffer,
                particleCount,
                deltaTime,
                Raylib.GetScreenWidth(),
                Raylib.GetScreenHeight()
            ));
        }

        private void SpawnParticles()
        {
            if (particleCount >= MaxParticles) return;
            var rand = new Random();
            
            // Temporary arrays for new particles
            int particlesToSpawn = 15;
            int countBefore = particleCount;
            int actualSpawn = 0;

            float[] newX = new float[particlesToSpawn];
            float[] newY = new float[particlesToSpawn];
            float[] newVelY = new float[particlesToSpawn];

            for (int k = 0; k < particlesToSpawn; k++) 
            {
                if (particleCount >= MaxParticles) break;

                newX[actualSpawn] = mousePosition.X + rand.Next(-10, 10);
                newY[actualSpawn] = mousePosition.Y + rand.Next(-10, 10);
                newVelY[actualSpawn] = 50; // Initial velocity

                particleCount++;
                actualSpawn++;
            }

            if (actualSpawn > 0)
            {
                posXBuffer.CopyFrom(newX, 0, countBefore, actualSpawn);
                posYBuffer.CopyFrom(newY, 0, countBefore, actualSpawn);
                velYBuffer.CopyFrom(newVelY, 0, countBefore, actualSpawn);
                
                // Ensure velX and acc are zero for new particles if needed?
                // The memory might be dirty from previous usage if we recycled index.
                // It's safer to clear.
                float[] zeros = new float[actualSpawn];
                velXBuffer.CopyFrom(zeros, 0, countBefore, actualSpawn);
                accXBuffer.CopyFrom(zeros, 0, countBefore, actualSpawn);
                accYBuffer.CopyFrom(zeros, 0, countBefore, actualSpawn);
            }
        }

        private void DrawGridDebug()
        {
            for (int x = 0; x < gridCols; x++)
                Raylib.DrawLine(x * GridCellSize, 0, x * GridCellSize, Raylib.GetScreenHeight(), new Color(30, 30, 30, 255));
            for (int y = 0; y < gridRows; y++)
                Raylib.DrawLine(0, y * GridCellSize, Raylib.GetScreenWidth(), y * GridCellSize, new Color(30, 30, 30, 255));
        }

        public void Dispose()
        {
            posXBuffer.Dispose();
            posYBuffer.Dispose();
            velXBuffer.Dispose();
            velYBuffer.Dispose();
            accXBuffer.Dispose();
            accYBuffer.Dispose();
            gridHeadsBuffer.Dispose();
            nextParticleBuffer.Dispose();
        }

        [Flags] private enum MouseButtons { None = 0, Left = 1, Right = 2 }
    }
}

// Shaders moved to global namespace to simplify Source Generator discovery


    [ComputeSharp.GeneratedComputeShaderDescriptor]
    [ComputeSharp.ThreadGroupSize(64, 1, 1)]
    public readonly partial struct ClearGridShaderV2 : IComputeShader
    {
        public readonly ReadWriteBuffer<int> gridHeads;

        public ClearGridShaderV2(ReadWriteBuffer<int> gridHeads)
        {
            this.gridHeads = gridHeads;
        }

        public void Execute()
        {
            gridHeads[ThreadIds.X] = -1;
        }
    }


    [ComputeSharp.GeneratedComputeShaderDescriptor]
    [ComputeSharp.ThreadGroupSize(64, 1, 1)]
    public readonly partial struct BuildGridShaderV2 : IComputeShader
    {
        public readonly ReadWriteBuffer<int> gridHeads;
        public readonly ReadWriteBuffer<int> nextParticle;
        public readonly ReadWriteBuffer<float> posX;
        public readonly ReadWriteBuffer<float> posY;
        public readonly int gridCols;
        public readonly int gridRows;
        public readonly int particleCount;
        public readonly int gridCellSize;

        public BuildGridShaderV2(
            ReadWriteBuffer<int> gridHeads,
            ReadWriteBuffer<int> nextParticle,
            ReadWriteBuffer<float> posX,
            ReadWriteBuffer<float> posY,
            int gridCols,
            int gridRows,
            int particleCount,
            int gridCellSize)
        {
            this.gridHeads = gridHeads;
            this.nextParticle = nextParticle;
            this.posX = posX;
            this.posY = posY;
            this.gridCols = gridCols;
            this.gridRows = gridRows;
            this.particleCount = particleCount;
            this.gridCellSize = gridCellSize;
        }

        public void Execute()
        {
            int i = ThreadIds.X;
            if (i >= particleCount) return;

            int cx = (int)(posX[i] / gridCellSize);
            int cy = (int)(posY[i] / gridCellSize);

            if (cx < 0) cx = 0; else if (cx >= gridCols) cx = gridCols - 1;
            if (cy < 0) cy = 0; else if (cy >= gridRows) cy = gridRows - 1;

            int cellIndex = cy * gridCols + cx;

            int originalHead;
            Hlsl.InterlockedExchange(ref gridHeads[cellIndex], i, out originalHead);
            nextParticle[i] = originalHead;
        }
    }


    [ComputeSharp.GeneratedComputeShaderDescriptor]
    [ComputeSharp.ThreadGroupSize(64, 1, 1)]
    public readonly partial struct CalculateForcesShaderV2 : IComputeShader
    {
        public readonly ReadWriteBuffer<float> accX;
        public readonly ReadWriteBuffer<float> accY;
        public readonly ReadWriteBuffer<float> posX;
        public readonly ReadWriteBuffer<float> posY;
        public readonly ReadWriteBuffer<float> velX;
        public readonly ReadWriteBuffer<float> velY;
        public readonly ReadWriteBuffer<int> gridHeads;
        public readonly ReadWriteBuffer<int> nextParticle;

        public readonly int gridCols;
        public readonly int gridRows;
        public readonly int particleCount;
        public readonly int gridCellSize;
        public readonly int width;
        public readonly int height;

        public readonly float wallMargin;
        public readonly float wallForce;
        public readonly float gravityY;
        public readonly float repulsionForce;
        public readonly float dampingFactor;
        public readonly float collisionRadius;
        public readonly float particleMass;

        // Mouse interactions
        public readonly int isMouseLeftDown;
        public readonly float mouseX;
        public readonly float mouseY;
        public readonly float mouseRadius;
        public readonly float mouseForce;

        public CalculateForcesShaderV2(
            ReadWriteBuffer<float> accX,
            ReadWriteBuffer<float> accY,
            ReadWriteBuffer<float> posX,
            ReadWriteBuffer<float> posY,
            ReadWriteBuffer<float> velX,
            ReadWriteBuffer<float> velY,
            ReadWriteBuffer<int> gridHeads,
            ReadWriteBuffer<int> nextParticle,
            int gridCols,
            int gridRows,
            int particleCount,
            int gridCellSize,
            int width,
            int height,
            float wallMargin,
            float wallForce,
            float gravityY,
            float repulsionForce,
            float dampingFactor,
            float collisionRadius,
            float particleMass,
            int isMouseLeftDown,
            float mouseX,
            float mouseY,
            float mouseRadius,
            float mouseForce)
        {
            this.accX = accX;
            this.accY = accY;
            this.posX = posX;
            this.posY = posY;
            this.velX = velX;
            this.velY = velY;
            this.gridHeads = gridHeads;
            this.nextParticle = nextParticle;
            this.gridCols = gridCols;
            this.gridRows = gridRows;
            this.particleCount = particleCount;
            this.gridCellSize = gridCellSize;
            this.width = width;
            this.height = height;
            this.wallMargin = wallMargin;
            this.wallForce = wallForce;
            this.gravityY = gravityY;
            this.repulsionForce = repulsionForce;
            this.dampingFactor = dampingFactor;
            this.collisionRadius = collisionRadius;
            this.particleMass = particleMass;
            this.isMouseLeftDown = isMouseLeftDown;
            this.mouseX = mouseX;
            this.mouseY = mouseY;
            this.mouseRadius = mouseRadius;
            this.mouseForce = mouseForce;
        }

        public void Execute()
        {
            int i = ThreadIds.X;
            if (i >= particleCount) return;

            float forceX = 0;
            float forceY = 0;

            float myX = posX[i];
            float myY = posY[i];

            // Wall Forces
            if (myX < wallMargin) forceX += (wallMargin - myX) * wallForce;
            else if (myX > width - wallMargin) forceX -= (myX - (width - wallMargin)) * wallForce;

            if (myY < wallMargin) forceY += (wallMargin - myY) * wallForce;
            else if (myY > height - wallMargin) forceY -= (myY - (height - wallMargin)) * wallForce;

            int cx = (int)(myX / gridCellSize);
            int cy = (int)(myY / gridCellSize);

            if (cx < 0) cx = 0; else if (cx >= gridCols) cx = gridCols - 1;
            if (cy < 0) cy = 0; else if (cy >= gridRows) cy = gridRows - 1;

            int startX = cx > 0 ? cx - 1 : 0;
            int endX = cx < gridCols - 1 ? cx + 1 : gridCols - 1;
            int startY = cy > 0 ? cy - 1 : 0;
            int endY = cy < gridRows - 1 ? cy + 1 : gridRows - 1;

            for (int y = startY; y <= endY; y++)
            {
                int rowOffset = y * gridCols;
                for (int x = startX; x <= endX; x++)
                {
                    int cellIndex = rowOffset + x;
                    int neighborIdx = gridHeads[cellIndex];

                    while (neighborIdx != -1)
                    {
                        if (i != neighborIdx)
                        {
                            float offX = myX - posX[neighborIdx];
                            float offY = myY - posY[neighborIdx];

                            if (Hlsl.Abs(offX) < collisionRadius && Hlsl.Abs(offY) < collisionRadius)
                            {
                                float distSqr = offX * offX + offY * offY;
                                if (distSqr < collisionRadius * collisionRadius && distSqr > 0.0001f)
                                {
                                    float distance = Hlsl.Sqrt(distSqr);
                                    float factor = (collisionRadius - distance) / distance * repulsionForce;

                                    forceX += offX * factor;
                                    forceY += offY * factor;

                                    float relVelX = velX[neighborIdx] - velX[i];
                                    float relVelY = velY[neighborIdx] - velY[i];
                                    forceX += relVelX * dampingFactor;
                                    forceY += relVelY * dampingFactor;
                                }
                            }
                        }
                        neighborIdx = nextParticle[neighborIdx];
                    }
                }
            }

            accX[i] = (forceX / particleMass);
            accY[i] = gravityY + (forceY / particleMass);

            if (isMouseLeftDown == 1)
            {
                float tmX = mouseX - myX;
                float tmY = mouseY - myY;
                float dSq = tmX * tmX + tmY * tmY;
                if (dSq < mouseRadius * mouseRadius)
                {
                    float dist = Hlsl.Sqrt(dSq);
                    float f = mouseForce / (dist + 1f);
                    accX[i] += tmX * f;
                    accY[i] += tmY * f;
                }
            }
        }
    }


    [ComputeSharp.GeneratedComputeShaderDescriptor]
    [ComputeSharp.ThreadGroupSize(64, 1, 1)]
    public readonly partial struct UpdateParticlesShaderV2 : IComputeShader
    {
        public readonly ReadWriteBuffer<float> posX;
        public readonly ReadWriteBuffer<float> posY;
        public readonly ReadWriteBuffer<float> velX;
        public readonly ReadWriteBuffer<float> velY;
        public readonly ReadWriteBuffer<float> accX;
        public readonly ReadWriteBuffer<float> accY;
        
        public readonly int particleCount;
        public readonly float deltaTime;
        public readonly int width;
        public readonly int height;

        public UpdateParticlesShaderV2(
            ReadWriteBuffer<float> posX,
            ReadWriteBuffer<float> posY,
            ReadWriteBuffer<float> velX,
            ReadWriteBuffer<float> velY,
            ReadWriteBuffer<float> accX,
            ReadWriteBuffer<float> accY,
            int particleCount,
            float deltaTime,
            int width,
            int height)
        {
            this.posX = posX;
            this.posY = posY;
            this.velX = velX;
            this.velY = velY;
            this.accX = accX;
            this.accY = accY;
            this.particleCount = particleCount;
            this.deltaTime = deltaTime;
            this.width = width;
            this.height = height;
        }
        
        public void Execute()
        {
            int i = ThreadIds.X;
            if (i >= particleCount) return;

            const float boundaryFriction = 0.5f;
            const float bounce = -0.2f;

            velX[i] += accX[i] * deltaTime;
            velY[i] += accY[i] * deltaTime;
            posX[i] += velX[i] * deltaTime;
            posY[i] += velY[i] * deltaTime;

            if (posX[i] < 0) { posX[i] = 0; velX[i] *= bounce; }
            if (posY[i] < 0) { posY[i] = 0; velY[i] *= bounce; }
            if (posX[i] > width) { posX[i] = width; velX[i] *= bounce; }
            if (posY[i] > height) { posY[i] = height; velY[i] *= bounce; velX[i] *= boundaryFriction; }
        }
    }

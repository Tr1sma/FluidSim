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
        private const int MaxParticles = 10000; //bis 7000 für Arbeit und bis 150000 für privat
        private int particleCount = 0;
        private const int InitialCount = 1500;

        private const float MouseForce = -2500f;
        private const float MouseRadius = 100f;
        private const int ParticlesToSpawn = 10;

        private const float WallMargin = 25f;
        private const float WallForce = 2000f + GravityY;

        private const float GravityY = 9.81f * 100f;

        // GPU Buffers
        private readonly ReadWriteBuffer<Float2> posBuffer;
        private readonly ReadWriteBuffer<Float2> velBuffer;
        private readonly ReadWriteBuffer<Float2> accBuffer;
        
        private readonly ReadWriteBuffer<int> gridHeadsBuffer;
        private readonly ReadWriteBuffer<int> nextParticleBuffer;

        // CPU arrays for rendering
        private readonly Float2[] cpuPos;
        
        private readonly int screenWidth;
        private readonly int screenHeight;
        
        // Raylib RenderTexture for GPU-accelerated drawing
        private RenderTexture2D targetTexture;
        
        private const float ParticleMass = 5f;
        private const float CollisionRadius = 10f;
        private const float RepulsionForce = 2000f;
        private const float DampingFactor = 10f;

        private const float PhysikHzRate = 100.0f;

        private const int GridCellSize = 10;
        private int gridCols;
        private int gridRows;

        private Vector2 mousePosition;
        private MouseButtons currentMouseButtons = MouseButtons.None;

        public Simulation()
        {
            screenWidth = Raylib.GetScreenWidth();
            screenHeight = Raylib.GetScreenHeight();

            gridCols = (screenWidth / GridCellSize) + 1; 
            gridRows = (screenHeight / GridCellSize) + 1;

            GraphicsDevice device = GraphicsDevice.GetDefault();
            
            posBuffer = device.AllocateReadWriteBuffer<Float2>(MaxParticles);
            velBuffer = device.AllocateReadWriteBuffer<Float2>(MaxParticles);
            accBuffer = device.AllocateReadWriteBuffer<Float2>(MaxParticles);
            
            gridHeadsBuffer = device.AllocateReadWriteBuffer<int>(gridCols * gridRows);
            nextParticleBuffer = device.AllocateReadWriteBuffer<int>(MaxParticles);

            cpuPos = new Float2[MaxParticles];
            
            // GPU-accelerated render target
            targetTexture = Raylib.LoadRenderTexture(screenWidth, screenHeight);

            InitializeParticles(screenWidth, screenHeight);
        }

        private void InitializeParticles(int width, int height)
        {
            var rand = new Random();
            Float2[] initPos = new Float2[InitialCount];
            Float2[] initVel = new Float2[InitialCount];

            for (int i = 0; i < InitialCount; i++)
            {
                initPos[i] = new Float2(
                    rand.Next(width / 2 - 100, width / 2 + 100),
                    rand.Next(height / 2 - 100, height / 2 + 100)
                );
            }

            posBuffer.CopyFrom(initPos, 0, 0, InitialCount);
            velBuffer.CopyFrom(initVel, 0, 0, InitialCount);
            accBuffer.CopyFrom(initVel, 0, 0, InitialCount);

            particleCount = InitialCount;
        }

        public void Run()
        {
            const float PhysicsStep = 1.0f / PhysikHzRate;
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

                if (particleCount > 0)
                {
                    posBuffer.CopyTo(cpuPos, 0, 0, particleCount);
                }

                Raylib.BeginTextureMode(targetTexture);
                Raylib.ClearBackground(Color.Black);

                for (int i = 0; i < particleCount; i++)
                {
                    Raylib.DrawPixel((int)cpuPos[i].X, (int)cpuPos[i].Y, Color.SkyBlue);
                }

                Raylib.EndTextureMode();

                Raylib.BeginDrawing();
                Raylib.DrawTextureRec(
                    targetTexture.Texture,
                    new Rectangle(0, 0, screenWidth, -screenHeight),
                    new Vector2(0, 0),
                    Color.White
                );

               
                Raylib.DrawRectangle(5, 5, 100, 25, new Color(0, 0, 0, 160));
                Raylib.DrawText($"FPS: {Raylib.GetFPS()}", 5, 5, 20, Color.Lime);
                Raylib.DrawText($"PhysSteps: {1.0f / PhysicsStep:F0} Hz", 5, 30, 10, Color.Gray);

                string countText = $"Particles: {particleCount}";
                Raylib.DrawRectangle(screenWidth - 160, 5, 150, 45, new Color(0, 0, 0, 160));
                Raylib.DrawText(countText, screenWidth - 155, 10, 20, Color.White);

                Raylib.EndDrawing();
            }
            Raylib.CloseWindow();
        }

        private void UpdateSimulation(float deltaTime)
        {
            if ((currentMouseButtons & MouseButtons.Right) != 0) 
            {
                SpawnParticles();
            }

            if (particleCount == 0) return;

            GraphicsDevice device = GraphicsDevice.GetDefault();

            int totalGridCells = gridCols * gridRows;
            device.For(totalGridCells, new ClearGridShaderOpt(gridHeadsBuffer));

            device.For(particleCount, new BuildGridShaderOpt(
                gridHeadsBuffer,
                nextParticleBuffer,
                posBuffer,
                gridCols,
                gridRows,
                particleCount,
                GridCellSize
            ));

            device.For(particleCount, new CalculateForcesShaderOpt(
                accBuffer,
                posBuffer,
                velBuffer,
                gridHeadsBuffer,
                nextParticleBuffer,
                gridCols,
                gridRows,
                particleCount,
                GridCellSize,
                screenWidth,
                screenHeight,
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

            device.For(particleCount, new UpdateParticlesShaderOpt(
                posBuffer,
                velBuffer,
                accBuffer,
                particleCount,
                deltaTime,
                screenWidth,
                screenHeight
            ));
        }

        private void SpawnParticles()
        {
            if (particleCount >= MaxParticles) return;
            var rand = new Random();
            
            int countBefore = particleCount;
            int actualSpawn = 0;

            Float2[] newPos = new Float2[ParticlesToSpawn];
            Float2[] newVel = new Float2[ParticlesToSpawn];

            for (int k = 0; k < ParticlesToSpawn; k++) 
            {
                if (particleCount >= MaxParticles) break;

                newPos[actualSpawn] = new Float2(
                    mousePosition.X + rand.Next(-10, 10),
                    mousePosition.Y + rand.Next(-10, 10)
                );
                newVel[actualSpawn] = new Float2(0, 50);

                particleCount++;
                actualSpawn++;
            }

            if (actualSpawn > 0)
            {
                posBuffer.CopyFrom(newPos, 0, countBefore, actualSpawn);
                velBuffer.CopyFrom(newVel, 0, countBefore, actualSpawn);
                
                Float2[] zeros = new Float2[actualSpawn];
                accBuffer.CopyFrom(zeros, 0, countBefore, actualSpawn);
            }
        }

        public void Dispose()
        {
            posBuffer.Dispose();
            velBuffer.Dispose();
            accBuffer.Dispose();
            gridHeadsBuffer.Dispose();
            nextParticleBuffer.Dispose();
            Raylib.UnloadRenderTexture(targetTexture);
        }

        [Flags] private enum MouseButtons { None = 0, Left = 1, Right = 2 }
    }
}

//256 for my RX 7900 XTX
[ComputeSharp.GeneratedComputeShaderDescriptor]
[ComputeSharp.ThreadGroupSize(16, 1, 1)]
public readonly partial struct ClearGridShaderOpt : IComputeShader
{
    public readonly ReadWriteBuffer<int> gridHeads;

    public ClearGridShaderOpt(ReadWriteBuffer<int> gridHeads)
    {
        this.gridHeads = gridHeads;
    }

    public void Execute()
    {
        gridHeads[ThreadIds.X] = -1;
    }
}


[ComputeSharp.GeneratedComputeShaderDescriptor]
[ComputeSharp.ThreadGroupSize(16, 1, 1)]
public readonly partial struct BuildGridShaderOpt : IComputeShader
{
    public readonly ReadWriteBuffer<int> gridHeads;
    public readonly ReadWriteBuffer<int> nextParticle;
    public readonly ReadWriteBuffer<Float2> pos;
    public readonly int gridCols;
    public readonly int gridRows;
    public readonly int particleCount;
    public readonly int gridCellSize;

    public BuildGridShaderOpt(
        ReadWriteBuffer<int> gridHeads,
        ReadWriteBuffer<int> nextParticle,
        ReadWriteBuffer<Float2> pos,
        int gridCols,
        int gridRows,
        int particleCount,
        int gridCellSize)
    {
        this.gridHeads = gridHeads;
        this.nextParticle = nextParticle;
        this.pos = pos;
        this.gridCols = gridCols;
        this.gridRows = gridRows;
        this.particleCount = particleCount;
        this.gridCellSize = gridCellSize;
    }

    public void Execute()
    {
        int i = ThreadIds.X;
        if (i >= particleCount) return;

        Float2 p = pos[i];
        int cx = (int)(p.X / gridCellSize);
        int cy = (int)(p.Y / gridCellSize);

        if (cx < 0) cx = 0; else if (cx >= gridCols) cx = gridCols - 1;
        if (cy < 0) cy = 0; else if (cy >= gridRows) cy = gridRows - 1;

        int cellIndex = cy * gridCols + cx;

        int originalHead;
        Hlsl.InterlockedExchange(ref gridHeads[cellIndex], i, out originalHead);
        nextParticle[i] = originalHead;
    }
}


[ComputeSharp.GeneratedComputeShaderDescriptor]
[ComputeSharp.ThreadGroupSize(16, 1, 1)]
public readonly partial struct CalculateForcesShaderOpt : IComputeShader
{
    public readonly ReadWriteBuffer<Float2> acc;
    public readonly ReadWriteBuffer<Float2> pos;
    public readonly ReadWriteBuffer<Float2> vel;
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

    public readonly int isMouseLeftDown;
    public readonly float mouseX;
    public readonly float mouseY;
    public readonly float mouseRadius;
    public readonly float mouseForce;

    public CalculateForcesShaderOpt(
        ReadWriteBuffer<Float2> acc,
        ReadWriteBuffer<Float2> pos,
        ReadWriteBuffer<Float2> vel,
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
        this.acc = acc;
        this.pos = pos;
        this.vel = vel;
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

        Float2 myPos = pos[i];
        Float2 myVel = vel[i];
        float myX = myPos.X;
        float myY = myPos.Y;

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

        float collisionRadiusSqr = collisionRadius * collisionRadius;

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
                        Float2 neighborPos = pos[neighborIdx];
                        float offX = myX - neighborPos.X;
                        float offY = myY - neighborPos.Y;

                        float distSqr = offX * offX + offY * offY;
                        if (distSqr < collisionRadiusSqr && distSqr > 0.0001f)
                        {
                            float distance = Hlsl.Sqrt(distSqr);
                            float factor = (collisionRadius - distance) / distance * repulsionForce;

                            forceX += offX * factor;
                            forceY += offY * factor;

                            Float2 neighborVel = vel[neighborIdx];
                            float relVelX = neighborVel.X - myVel.X;
                            float relVelY = neighborVel.Y - myVel.Y;
                            forceX += relVelX * dampingFactor;
                            forceY += relVelY * dampingFactor;
                        }
                    }
                    neighborIdx = nextParticle[neighborIdx];
                }
            }
        }

        float accX = forceX / particleMass;
        float accY = gravityY + (forceY / particleMass);

        if (isMouseLeftDown == 1)
        {
            float tmX = mouseX - myX;
            float tmY = mouseY - myY;
            float dSq = tmX * tmX + tmY * tmY;
            if (dSq < mouseRadius * mouseRadius)
            {
                float dist = Hlsl.Sqrt(dSq);
                float f = mouseForce / (dist + 1f);
                accX += tmX * f;
                accY += tmY * f;
            }
        }

        acc[i] = new Float2(accX, accY);
    }
}


[ComputeSharp.GeneratedComputeShaderDescriptor]
[ComputeSharp.ThreadGroupSize(16, 1, 1)]
public readonly partial struct UpdateParticlesShaderOpt : IComputeShader
{
    public readonly ReadWriteBuffer<Float2> pos;
    public readonly ReadWriteBuffer<Float2> vel;
    public readonly ReadWriteBuffer<Float2> acc;
    
    public readonly int particleCount;
    public readonly float deltaTime;
    public readonly int width;
    public readonly int height;

    public UpdateParticlesShaderOpt(
        ReadWriteBuffer<Float2> pos,
        ReadWriteBuffer<Float2> vel,
        ReadWriteBuffer<Float2> acc,
        int particleCount,
        float deltaTime,
        int width,
        int height)
    {
        this.pos = pos;
        this.vel = vel;
        this.acc = acc;
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

        Float2 v = vel[i];
        Float2 a = acc[i];
        Float2 p = pos[i];

        v.X += a.X * deltaTime;
        v.Y += a.Y * deltaTime;
        p.X += v.X * deltaTime;
        p.Y += v.Y * deltaTime;

        if (p.X < 0) { p.X = 0; v.X *= bounce; }
        if (p.Y < 0) { p.Y = 0; v.Y *= bounce; }
        if (p.X > width) { p.X = width; v.X *= bounce; }
        if (p.Y > height) { p.Y = height; v.Y *= bounce; v.X *= boundaryFriction; }

        vel[i] = v;
        pos[i] = p;
    }
}

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using Raylib_cs;

namespace FluidSimulation
{
    public class Program
    {
        public static void Main()
        {
            Raylib.InitWindow(800, 450, "Fluid Simulation");
            Raylib.SetTargetFPS(9999); 

            var sim = new Simulation();
            sim.Run();
        }
    }

        private static void OnUpdate(double deltaTime)
        {
            sim.Update((float)deltaTime);
        }

        private static void OnRender(double deltaTime)
        {
            sim.Render((float)deltaTime);
        }

        private static void OnClosing()
        {
            sim.Dispose();
        }
    }

    public unsafe class Simulation : IDisposable
    {
        private GL Gl;
        private IInputContext inputContext;
        private ImGuiController imGuiController;
        private IWindow window;

        // OpenGL resources
        private uint vao;
        private uint vbo;
        private uint shaderProgram;

        private const int MaxParticles = 10000;
        private static int particleCount = 0;
        private const int InitialCount = 1500;

        private float mouseForce = -3500f + (particleCount / 10);
        private const float MouseRadius = 100f;
        private const int ParticlesToSpawn = 10;

        private const float WallMargin = 25f;
        private const float WallForce = 2000f + GravityY;
        private const float GravityY = 9.81f * 100f;

        // GPU Buffers (ComputeSharp)
        private ReadWriteBuffer<Float2> posBuffer;
        private ReadWriteBuffer<Float2> velBuffer;
        private ReadWriteBuffer<Float2> accBuffer;
        
        private ReadWriteBuffer<int> gridHeadsBuffer;
        private ReadWriteBuffer<int> nextParticleBuffer;

        // CPU arrays for rendering
        private readonly Float2[] cpuPos;
        
        private int screenWidth;
        private int screenHeight;
        
        private const float ParticleMass = 5f;
        private const float MouseForce = -1000f;
        private const float MouseRadius = 100f;
        private const float CollisionRadius = 10f;
        private const float RepulsionForce = 2000f;
        private const float DampingFactor = 10f;
        private const float WallMargin = 20;
        private const float WallForce = 200f + GravityY;

        private const float GravityY = 45f * 10f;
        //9.81 gewichtskraft erde
        //45 f�r wasser �hnliches verhalten

        private float[] posX = new float[MaxParticles];
        private float[] posY = new float[MaxParticles];
        private float[] velX = new float[MaxParticles];
        private float[] velY = new float[MaxParticles];
        private float[] accX = new float[MaxParticles];
        private float[] accY = new float[MaxParticles];



        private const int GridCellSize = 12; // >= CollisionRadius NICHT KLEINER SONST KRACHTS!!!!!!!!!
        private int gridCols;
        private int gridRows;
        private int[] gridHeads;      
        private int[] nextParticle;

        private Vector2 mousePosition;
        private MouseButtons currentMouseButtons = MouseButtons.None;

        public Simulation()
        {
            InitializeGrid();
            InitializeParticles();
        }

        private void InitializeGrid()
        {
            int width = Raylib.GetScreenWidth();
            int height = Raylib.GetScreenHeight();

            gridCols = (width / GridCellSize) + 1; 
            gridRows = (height / GridCellSize) + 1;

            gridHeads = new int[gridCols * gridRows];
            nextParticle = new int[MaxParticles];
        }

        public void Run()
        {
            while (!Raylib.WindowShouldClose())
            {
                mousePosition = Raylib.GetMousePosition();
                currentMouseButtons = MouseButtons.None;
                if (Raylib.IsMouseButtonDown(MouseButton.Left)) currentMouseButtons |= MouseButtons.Left;
                if (Raylib.IsMouseButtonDown(MouseButton.Right)) currentMouseButtons |= MouseButtons.Right;

                UpdateSimulation();

                Raylib.BeginDrawing();
                Raylib.ClearBackground(Color.Black);

                //DrawGridDebug(); 

                for (int i = 0; i < particleCount; i++)
                {
                    Raylib.DrawCircleV(new Vector2(posX[i], posY[i]), 4, Color.SkyBlue);
                }

                Raylib.DrawRectangle(5, 5, 100, 25, new Color(0, 0, 0, 160));
                Raylib.DrawText($"FPS: {Raylib.GetFPS()}", 5, 5, 20, Color.Lime);

                string countText = $"Particles: {particleCount}";
                Raylib.DrawRectangle(Raylib.GetScreenWidth() - 160, 5, 150, 45, new Color(0, 0, 0, 160));
                Raylib.DrawText(countText, Raylib.GetScreenWidth() - 155, 10, 20, Color.White);

                Raylib.EndDrawing();
            }
            Raylib.CloseWindow();
        }

        private void InitializeParticles()
        {
            var rand = new Random();
            int width = Raylib.GetScreenWidth();
            int height = Raylib.GetScreenHeight();

            for (int i = 0; i < InitialCount; i++)
            {
                if (particleCount >= MaxParticles) break;
                posX[particleCount] = rand.Next(width / 2 - 100, width / 2 + 100);
                posY[particleCount] = rand.Next(height / 2 - 100, height / 2 + 100);
                particleCount++;
            }
        }

        private void UpdateSimulation()
        {
            float deltaTime = Raylib.GetFrameTime();
            if (deltaTime > 0.1f) deltaTime = 0.1f;

            if ((currentMouseButtons & MouseButtons.Right) != 0) SpawnParticles();

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
                isLeftMouseDown ? 1 : 0,
                mousePosition.X,
                mousePosition.Y,
                MouseRadius,
                mouseForce
            ));

            device.For(particleCount, new UpdateParticlesShaderOpt(
                posBuffer,
                velBuffer,
                accBuffer,
                particleCount,
                dt,
                screenWidth,
                screenHeight
            ));
        }

        private void SpawnParticles()
        {
            if (particleCount >= MaxParticles) return;
            var rand = new Random();
            for (int k = 0; k < 5; k++) 
            {
                if (particleCount >= MaxParticles) return;
                posX[particleCount] = mousePosition.X + rand.Next(-10, 10);
                posY[particleCount] = mousePosition.Y + rand.Next(-10, 10);
                velY[particleCount] = 50;
                particleCount++;
            }
        }

        private void UpdateGrid()
        {
            // Reset grid heads. -1 is empty
            Array.Fill(gridHeads, -1);

           for(int i = 0; i < particleCount; i ++)
           {
                //Cell Koordinaten
                int cx = (int)(posX[i] / GridCellSize);
                int cy = (int)(posY[i] / GridCellSize);

                if (cx < 0) cx = 0; else if (cx >= gridCols) cx = gridCols - 1;
                if (cy < 0) cy = 0; else if (cy >= gridRows) cy = gridRows - 1;

                int cellIndex = cy * gridCols + cx;

                //aktuellr Head wird zum Next dieses Partikels
                nextParticle[i] = gridHeads[cellIndex];
                //Partikel wird neuer Head der zelle
                gridHeads[cellIndex] = i;
           };
        }

        private unsafe void CalculateForces()
        {
            int width = Raylib.GetScreenWidth();
            int height = Raylib.GetScreenHeight();

            fixed (float* pPosX = posX, pPosY = posY, pVelX = velX, pVelY = velY, pAccX = accX, pAccY = accY)
            fixed (int* pGridHeads = gridHeads, pNextParticle = nextParticle)
            {
                long addrPosX = (long)pPosX;
                long addrPosY = (long)pPosY;
                long addrVelX = (long)pVelX;
                long addrVelY = (long)pVelY;
                long addrAccX = (long)pAccX;
                long addrAccY = (long)pAccY;

                long addrGridHeads = (long)pGridHeads;
                long addrNextParticle = (long)pNextParticle;

                Parallel.For(0, particleCount, i =>
                {
                    float* lPosX = (float*)addrPosX;
                    float* lPosY = (float*)addrPosY;
                    float* lVelX = (float*)addrVelX;
                    float* lVelY = (float*)addrVelY;
                    float* lAccX = (float*)addrAccX;
                    float* lAccY = (float*)addrAccY;

                    int* lGridHeads = (int*)addrGridHeads;
                    int* lNextParticle = (int*)addrNextParticle;

                    float forceX = 0;
                    float forceY = 0;

                    float myX = lPosX[i];
                    float myY = lPosY[i];

                    if (myX < WallMargin) forceX += (WallMargin - myX) * WallForce;
                    else if (myX > width - WallMargin) forceX -= (myX - (width - WallMargin)) * WallForce;

                    if (myY < WallMargin) forceY += (WallMargin - myY) * WallForce;
                    else if (myY > height - WallMargin) forceY -= (myY - (height - WallMargin)) * WallForce;

                    int cx = (int)(myX / GridCellSize);
                    int cy = (int)(myY / GridCellSize);

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
                            int neighborIdx = lGridHeads[cellIndex];

                            while (neighborIdx != -1)
                            {
                                if (i != neighborIdx)
                                {
                                    float offX = myX - lPosX[neighborIdx];
                                    float offY = myY - lPosY[neighborIdx];

                                    if (Math.Abs(offX) < CollisionRadius && Math.Abs(offY) < CollisionRadius)
                                    {
                                        float distSqr = offX * offX + offY * offY;
                                        if (distSqr < CollisionRadius * CollisionRadius && distSqr > 0.0001f)
                                        {
                                            float distance = MathF.Sqrt(distSqr);
                                            float factor = (CollisionRadius - distance) / distance * RepulsionForce;

                                            forceX += offX * factor;
                                            forceY += offY * factor;

                                            float relVelX = lVelX[neighborIdx] - lVelX[i];
                                            float relVelY = lVelY[neighborIdx] - lVelY[i];
                                            forceX += relVelX * DampingFactor;
                                            forceY += relVelY * DampingFactor;
                                        }
                                    }
                                }
                                neighborIdx = lNextParticle[neighborIdx];
                            }
                        }
                    }

                    lAccX[i] = (forceX / ParticleMass);
                    lAccY[i] = GravityY + (forceY / ParticleMass);

                    if ((currentMouseButtons & MouseButtons.Left) != 0)
                    {
                        float tmX = mousePosition.X - posX[i];
                        float tmY = mousePosition.Y - posY[i];
                        float dSq = tmX * tmX + tmY * tmY;
                        if (dSq < MouseRadius * MouseRadius)
                        {
                            float dist = MathF.Sqrt(dSq);
                            float f = MouseForce / (dist + 1f);
                            accX[i] += tmX * f;
                            accY[i] += tmY * f;
                        }
                    }
                });
            }
        }

        private void UpdateParticles(float deltaTime)
        {
            const float boundaryFriction = 0.5f;
            const float bounce = -0.2f;
            int width = Raylib.GetScreenWidth();
            int height = Raylib.GetScreenHeight();

            Parallel.For(0, particleCount, i =>
            {
                velX[i] += accX[i] * deltaTime;
                velY[i] += accY[i] * deltaTime;
                posX[i] += velX[i] * deltaTime;
                posY[i] += velY[i] * deltaTime;

                if (posX[i] < 0) { posX[i] = 0; velX[i] *= bounce; }
                if (posY[i] < 0) { posY[i] = 0; velY[i] *= bounce; }
                if (posX[i] > width) { posX[i] = width; velX[i] *= bounce; }
                if (posY[i] > height) { posY[i] = height; velY[i] *= bounce; velX[i] *= boundaryFriction; }
            });
        }

        private void DrawGridDebug()
        {
            for (int x = 0; x < gridCols; x++)
                Raylib.DrawLine(x * GridCellSize, 0, x * GridCellSize, Raylib.GetScreenHeight(), new Color(30, 30, 30, 255));
            for (int y = 0; y < gridRows; y++)
                Raylib.DrawLine(0, y * GridCellSize, Raylib.GetScreenWidth(), y * GridCellSize, new Color(30, 30, 30, 255));
        }

        [Flags] private enum MouseButtons { None = 0, Left = 1, Right = 2 }
    }
}
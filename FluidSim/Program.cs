using System;
using System.Numerics;
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

    public class Simulation
    {
        private const int MaxParticles = 4000; 
        private int particleCount = 0;
        private const int InitialCount = 500;

        private const float ParticleMass = 5f;
        private const float MouseForce = -1000f;
        private const float MouseRadius = 100f;
        private const float CollisionRadius = 10f;
        private const float RepulsionForce = 2000f;
        private const float DampingFactor = 10f;
        private const float WallMargin = 20;
        private const float WallForce = 200f;
        private const float GravityY = 9.81f * 10f;

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

            UpdateGrid();
            CalculateForces();
            UpdateParticles(deltaTime);
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

            Parallel.For(0, particleCount, i =>
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
            });
        }

        private void CalculateForces()
        {
            int width = Raylib.GetScreenWidth();
            int height = Raylib.GetScreenHeight();

            Parallel.For(0, particleCount, i =>
            {
                float forceX = 0;
                float forceY = 0;

                if (posX[i] < WallMargin) forceX += (WallMargin - posX[i]) * WallForce;
                if (posX[i] > width - WallMargin) forceX -= (posX[i] - (width - WallMargin)) * WallForce;
                if (posY[i] < WallMargin) forceY += (WallMargin - posY[i]) * WallForce;
                if (posY[i] > height - WallMargin) forceY -= (posY[i] - (height - WallMargin)) * WallForce;

                int cx = (int)(posX[i] / GridCellSize);
                int cy = (int)(posY[i] / GridCellSize);

                int startX = Math.Max(0, cx - 1);
                int endX = Math.Min(gridCols - 1, cx + 1);
                int startY = Math.Max(0, cy - 1);
                int endY = Math.Min(gridRows - 1, cy + 1);

                for (int y = startY; y <= endY; y++)
                {
                    for (int x = startX; x <= endX; x++)
                    {
                        int cellIndex = y * gridCols + x;

                        int neighborIdx = gridHeads[cellIndex];
                        while (neighborIdx != -1)
                        {
                            if (i != neighborIdx)
                            {
                                float offX = posX[i] - posX[neighborIdx];
                                float offY = posY[i] - posY[neighborIdx];

                                if (Math.Abs(offX) < CollisionRadius && Math.Abs(offY) < CollisionRadius)
                                {
                                    float distSqr = offX * offX + offY * offY;
                                    if (distSqr < CollisionRadius * CollisionRadius && distSqr > 0.0001f)
                                    {
                                        float distance = MathF.Sqrt(distSqr);
                                        float dirX = offX / distance;
                                        float dirY = offY / distance;
                                        float overlap = CollisionRadius - distance;

                                        forceX += dirX * overlap * RepulsionForce;
                                        forceY += dirY * overlap * RepulsionForce;

                                        float relVelX = velX[neighborIdx] - velX[i];
                                        float relVelY = velY[neighborIdx] - velY[i];
                                        forceX += relVelX * DampingFactor;
                                        forceY += relVelY * DampingFactor;
                                    }
                                }
                            }
                            neighborIdx = nextParticle[neighborIdx];
                        }
                    }
                }

                accX[i] = (forceX / ParticleMass);
                accY[i] = GravityY + (forceY / ParticleMass);

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
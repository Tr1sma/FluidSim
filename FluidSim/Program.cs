using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using Raylib_cs;

namespace FluidSimulation
{
    public class Program
    {
        public static void Main()
        {
            Raylib.InitWindow(800, 450, "Fluid Simulation");
            Raylib.SetTargetFPS(60);

            var sim = new Simulation();
            sim.Run();
        }
    }

    public class Simulation
    {
        private const int ParticleCount = 500;
        private const float ParticleMass = 5f;
        private const float MouseForce = -1000f;
        private const float MouseRadius = 100f;

        private const float CollisionRadius = 10f;
        private const float RepulsionForce = 2000f;
        private const float DampingFactor = 10f;
        private const float WallMargin = 20f;
        private const float WallForce = 20f;

        private readonly List<Particle> particles = new List<Particle>();
        private Vector2 mousePosition;
        private MouseButtons currentMouseButtons = MouseButtons.None;

        private readonly Stopwatch stopwatch = new Stopwatch();
        private double frameCount = 0;
        private double currentFps = 0;
        private double timeSinceLastFpsUpdate = 0;
        private double lastFrameTime = 0;
        private Vector2 gravity = new(0, 9.81f * 10f);

        public Simulation()
        {
            InitializeParticles();
            stopwatch.Start();
        }

        public void Run()
        {
            while (!Raylib.WindowShouldClose())
            {
                // Input
                mousePosition = Raylib.GetMousePosition();
                currentMouseButtons = MouseButtons.None;
                if (Raylib.IsMouseButtonDown(MouseButton.Left)) currentMouseButtons |= MouseButtons.Left;
                if (Raylib.IsMouseButtonDown(MouseButton.Right)) currentMouseButtons |= MouseButtons.Right;

                // Update
                UpdateSimulation();

                // Draw
                Raylib.BeginDrawing();
                Raylib.ClearBackground(Color.Black);

                foreach (var particle in particles)
                {
                    Raylib.DrawCircleV(particle.Position, 4, Color.SkyBlue);
                }

                // UI
                Raylib.DrawRectangle(5, 5, 100, 25, new Color(0, 0, 0, 160));
                Raylib.DrawText($"FPS: {currentFps:F1}", 10, 10, 20, Color.Lime);

                string countText = $"Particles: {particles.Count}";
                string helpText = "L-Click: Attract | R-Click: Spawn";

                int screenWidth = Raylib.GetScreenWidth();
                int panelW = 220;
                int panelX = screenWidth - panelW - 5;

                Raylib.DrawRectangle(panelX, 5, panelW, 45, new Color(0, 0, 0, 160));
                Raylib.DrawText(countText, panelX + 5, 10, 20, Color.White);
                Raylib.DrawText(helpText, panelX + 5, 25, 20, Color.Orange);

                Raylib.EndDrawing();
            }

            Raylib.CloseWindow();
        }

        private void InitializeParticles()
        {
            var rand = new Random();
            int width = Raylib.GetScreenWidth();
            int height = Raylib.GetScreenHeight();

            for (int i = 0; i < ParticleCount; i++)
            {
                particles.Add(new Particle
                {
                    Position = new Vector2(
                        rand.Next(width / 2 - 50, width / 2 + 50),
                        rand.Next(height / 2 - 50, height / 2 + 50)),
                    Velocity = Vector2.Zero
                });
            }
        }

        private void UpdateSimulation()
        {
            double currentTime = stopwatch.Elapsed.TotalSeconds;
            float deltaTime = (float)(currentTime - lastFrameTime);
            lastFrameTime = currentTime;

            frameCount++;
            timeSinceLastFpsUpdate += deltaTime;
            if (timeSinceLastFpsUpdate >= 1.0)
            {
                currentFps = frameCount / timeSinceLastFpsUpdate;
                frameCount = 0;
                timeSinceLastFpsUpdate = 0;
            }

            if (deltaTime > 0.1f) deltaTime = 0.1f;

            if ((currentMouseButtons & MouseButtons.Right) != 0)
            {
                SpawnParticles();
            }

            CalculateForces();
            UpdateParticles(deltaTime);
        }

        private void SpawnParticles()
        {
            var rand = new Random();
            for (int i = 0; i < 5; i++)
            {
                particles.Add(new Particle
                {
                    Position = mousePosition + new Vector2(
                        rand.Next(-10, 10),
                        rand.Next(-10, 10)),
                    Velocity = new Vector2(rand.Next(-20, 20), 50)
                });
            }
        }

        private void CalculateForces()
        {
            int width = Raylib.GetScreenWidth();
            int height = Raylib.GetScreenHeight();

            foreach (var particle in particles)
            {
                Vector2 collisionForce = Vector2.Zero;

                // Wandabstoßung berechnen
                if (particle.Position.X < WallMargin)
                {
                    collisionForce.X += (WallMargin - particle.Position.X) * WallForce;
                }
                if (particle.Position.X > width - WallMargin)
                {
                    collisionForce.X -= (particle.Position.X - (width - WallMargin)) * WallForce;
                }
                if (particle.Position.Y < WallMargin)
                {
                    collisionForce.Y += (WallMargin - particle.Position.Y) * WallForce;
                }
                if (particle.Position.Y > height - WallMargin)
                {
                    collisionForce.Y -= (particle.Position.Y - (height - WallMargin)) * WallForce;
                }

                foreach (var other in particles)
                {
                    if (particle == other) continue;

                    Vector2 offset = particle.Position - other.Position;
                    float distance = offset.Length();

                    if (distance < CollisionRadius && distance > 0)
                    {
                        Vector2 dir = offset / distance;
                        float overlap = CollisionRadius - distance;
                        collisionForce += dir * overlap * RepulsionForce;

                        Vector2 relVel = other.Velocity - particle.Velocity;
                        collisionForce += relVel * DampingFactor;
                    }
                }

                particle.Acceleration = gravity + (collisionForce / ParticleMass);

                ApplyMouseInteraction(particle);
            }
        }

        private void ApplyMouseInteraction(Particle particle)
        {
            if ((currentMouseButtons & MouseButtons.Left) != 0)
            {
                Vector2 toMouse = mousePosition - particle.Position;
                float distance = toMouse.Length();
                if (distance < MouseRadius)
                {
                    particle.Acceleration += toMouse * MouseForce / (distance + 1f);
                }
            }
        }

        private void UpdateParticles(float deltaTime)
        {
            const float boundaryFriction = 0.5f;
            const float bounce = -0.2f;

            int width = Raylib.GetScreenWidth();
            int height = Raylib.GetScreenHeight();

            foreach (var particle in particles)
            {
                particle.Velocity += particle.Acceleration * deltaTime;
                particle.Position += particle.Velocity * deltaTime;

                if (particle.Position.X < 0)
                {
                    particle.Position.X = 0;
                    particle.Velocity.X *= bounce;
                }
                if (particle.Position.Y < 0)
                {
                    particle.Position.Y = 0;
                    particle.Velocity.Y *= bounce;
                }
                if (particle.Position.X > width)
                {
                    particle.Position.X = width;
                    particle.Velocity.X *= bounce;
                }
                if (particle.Position.Y > height)
                {
                    particle.Position.Y = height;
                    particle.Velocity.Y *= bounce;
                    particle.Velocity.X *= boundaryFriction;
                }
            }
        }

        private class Particle
        {
            public Vector2 Position;
            public Vector2 Velocity;
            public Vector2 Acceleration;
        }

        [Flags]
        private enum MouseButtons
        {
            None = 0,
            Left = 1,
            Right = 2
        }
    }
}

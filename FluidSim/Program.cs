using System;
using System.Drawing;
using System.Numerics;
using System.Runtime.InteropServices;
using ComputeSharp;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.Windowing;
using Silk.NET.OpenGL.Extensions.ImGui;
using ImGuiNET;

namespace FluidSimulation
{
    public class Program
    {
        private static IWindow window;
        private static Simulation sim;

        public static void Main()
        {
            var options = WindowOptions.Default;
            options.Size = new Vector2D<int>(800, 450);
            options.Title = "Fluid Simulation";
            options.VSync = false;

            window = Window.Create(options);

            window.Load += OnLoad;
            window.Update += OnUpdate;
            window.Render += OnRender;
            window.Closing += OnClosing;

            window.Run();
        }

        private static void OnLoad()
        {
            sim = new Simulation(window);
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
        private int particleCount = 0;
        private const int InitialCount = 1500;

        private const float MouseForce = -2500f;
        private const float MouseRadius = 100f;
        private const int ParticlesToSpawn = 10;

        private const float WallMargin = 25f;
        private const float WallForce = 200 + GravityY;
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
        private const float CollisionRadius = 10f;
        private const float RepulsionForce = 2000f;
        private const float DampingFactor = 10f;

        private const float PhysikHzRate = 66.0f;

        private const int GridCellSize = 10;
        private int gridCols;
        private int gridRows;

        private Vector2 mousePosition;
        private bool isLeftMouseDown;
        private bool isRightMouseDown;

        public Simulation(IWindow window)
        {
            this.window = window;
            this.screenWidth = window.Size.X;
            this.screenHeight = window.Size.Y;

            // Init OpenGL
            Gl = window.CreateOpenGL();
            inputContext = window.CreateInput();
            imGuiController = new ImGuiController(Gl, window, inputContext);

            // Register resize event
            window.Resize += OnResize;

            // Init Simulation Data
            gridCols = (screenWidth / GridCellSize) + 1;
            gridRows = (screenHeight / GridCellSize) + 1;

            GraphicsDevice device = GraphicsDevice.GetDefault();
            
            posBuffer = device.AllocateReadWriteBuffer<Float2>(MaxParticles);
            velBuffer = device.AllocateReadWriteBuffer<Float2>(MaxParticles);
            accBuffer = device.AllocateReadWriteBuffer<Float2>(MaxParticles);
            
            gridHeadsBuffer = device.AllocateReadWriteBuffer<int>(gridCols * gridRows);
            nextParticleBuffer = device.AllocateReadWriteBuffer<int>(MaxParticles);

            cpuPos = new Float2[MaxParticles];

            InitializeParticles(screenWidth, screenHeight);
            InitGraphics();

        }

        private void OnResize(Vector2D<int> newSize)
        {
            screenWidth = newSize.X;
            screenHeight = newSize.Y;
            Gl.Viewport(newSize);
            
            // Re-calc grid if needed, or just keep it fixed to max size? 
            // For simplicity, we keep buffers as is, but simulation boundaries update.
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

        private void InitGraphics()
        {
            // Create VAO
            vao = Gl.GenVertexArray();
            Gl.BindVertexArray(vao);

            // Create VBO
            vbo = Gl.GenBuffer();
            Gl.BindBuffer(BufferTargetARB.ArrayBuffer, vbo);
            // Pre-allocate buffer for max particles
            Gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(MaxParticles * sizeof(Float2)), null, BufferUsageARB.DynamicDraw);

            // Vertex Attribute (Pos)
            Gl.EnableVertexAttribArray(0);
            Gl.VertexAttribPointer(0, 2, VertexAttribPointerType.Float, false, (uint)sizeof(Float2), (void*)0);

            // Compile Shaders
            string vertexSource = @"
                #version 330 core
                layout (location = 0) in vec2 aPos;
                uniform mat4 uProjection;
                void main()
                {
                    gl_Position = uProjection * vec4(aPos, 0.0, 1.0);
                }
                ";
            string fragmentSource = @"
                #version 330 core
                out vec4 FragColor;
                void main()
                {
                    FragColor = vec4(0.53, 0.81, 0.92, 1.0); // SkyBlue
                }
                ";
            shaderProgram = CreateShaderProgram(vertexSource, fragmentSource);
        }

        private uint CreateShaderProgram(string vertexSrc, string fragSrc)
        {
            uint vs = Gl.CreateShader(ShaderType.VertexShader);
            Gl.ShaderSource(vs, vertexSrc);
            Gl.CompileShader(vs);
            CheckShader(vs);

            uint fs = Gl.CreateShader(ShaderType.FragmentShader);
            Gl.ShaderSource(fs, fragSrc);
            Gl.CompileShader(fs);
            CheckShader(fs);

            uint prog = Gl.CreateProgram();
            Gl.AttachShader(prog, vs);
            Gl.AttachShader(prog, fs);
            Gl.LinkProgram(prog);
            
            Gl.DeleteShader(vs);
            Gl.DeleteShader(fs);
            return prog;
        }

        private void CheckShader(uint shader)
        {
            string infoLog = Gl.GetShaderInfoLog(shader);
            if (!string.IsNullOrWhiteSpace(infoLog))
            {
                Console.WriteLine($"Shader Compile Error: {infoLog}");
            }
        }

        private double accumulator = 0.0;
        private const float PhysicsStep = 1.0f / PhysikHzRate;

        public void Update(float deltaTime)
        {
            // Input
            if (inputContext.Mice.Count > 0)
            {
                var mouse = inputContext.Mice[0];
                mousePosition = new Vector2(mouse.Position.X, mouse.Position.Y);
                isLeftMouseDown = mouse.IsButtonPressed(MouseButton.Left);
                isRightMouseDown = mouse.IsButtonPressed(MouseButton.Right);
            }

            if (deltaTime > 0.25f) deltaTime = 0.25f;

            accumulator += deltaTime;

            while (accumulator >= PhysicsStep)
            {
                UpdateSimulation(PhysicsStep);
                accumulator -= PhysicsStep;
            }

            // Sync to CPU for rendering
            if (particleCount > 0)
            {
                posBuffer.CopyTo(cpuPos, 0, 0, particleCount);
            }
            
            // Update ImGui
            imGuiController.Update(deltaTime);
        }

        public void Render(float deltaTime)
        {
            Gl.ClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            Gl.Clear(ClearBufferMask.ColorBufferBit);

            if (particleCount > 0)
            {
                Gl.UseProgram(shaderProgram);
                
                // Set Projection Matrix (Ortho top-left)
                var projection = Matrix4x4.CreateOrthographicOffCenter(0, screenWidth, screenHeight, 0, -1, 1);
                int loc = Gl.GetUniformLocation(shaderProgram, "uProjection");
                Gl.UniformMatrix4(loc, 1, false, (float*)&projection);

                Gl.BindVertexArray(vao);
                Gl.BindBuffer(BufferTargetARB.ArrayBuffer, vbo);
                
                // Update VBO data
                fixed (void* data = cpuPos)
                {
                    Gl.BufferSubData(BufferTargetARB.ArrayBuffer, 0, (nuint)(particleCount * sizeof(Float2)), data);
                }

                // Draw
                Gl.DrawArrays(PrimitiveType.Points, 0, (uint)particleCount);
            }

            // Draw ImGui
            ImGui.SetNextWindowPos(new Vector2(5, 5), ImGuiCond.Always);
            ImGui.SetNextWindowBgAlpha(0.6f);
            ImGui.Begin("Stats", ImGuiWindowFlags.NoDecoration | ImGuiWindowFlags.AlwaysAutoResize | ImGuiWindowFlags.NoSavedSettings | ImGuiWindowFlags.NoFocusOnAppearing | ImGuiWindowFlags.NoNav);
            ImGui.TextColored(new Vector4(0, 1, 0, 1), $"FPS: {ImGui.GetIO().Framerate:F1}");
            ImGui.TextColored(new Vector4(0.5f, 0.5f, 0.5f, 1), $"PhysSteps: {PhysikHzRate:F0} Hz");
            ImGui.End();

            ImGui.SetNextWindowPos(new Vector2(screenWidth - 160, 5), ImGuiCond.Always);
            ImGui.SetNextWindowBgAlpha(0.6f);
            ImGui.Begin("Controls", ImGuiWindowFlags.NoDecoration | ImGuiWindowFlags.AlwaysAutoResize | ImGuiWindowFlags.NoSavedSettings | ImGuiWindowFlags.NoFocusOnAppearing | ImGuiWindowFlags.NoNav);
            ImGui.Text($"Particles: {particleCount}");
            ImGui.End();

            imGuiController.Render();
        }

        private void UpdateSimulation(float dt)
        {
            if (isRightMouseDown) 
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
                isLeftMouseDown ? 1 : 0,
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
                dt,
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
            
            imGuiController?.Dispose();
            inputContext?.Dispose();
            Gl?.Dispose();
        }
    }

    [ComputeSharp.GeneratedComputeShaderDescriptor]
    [ComputeSharp.ThreadGroupSize(32, 1, 1)]
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
    [ComputeSharp.ThreadGroupSize(32, 1, 1)]
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
    [ComputeSharp.ThreadGroupSize(32, 1, 1)]
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
    [ComputeSharp.ThreadGroupSize(32, 1, 1)]
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
}
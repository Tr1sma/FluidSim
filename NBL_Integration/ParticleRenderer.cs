using System;
using System.Numerics;
using Silk.NET.OpenGL;
using ComputeSharp;

namespace NBL.Particles
{
    public unsafe class ParticleRenderer : IDisposable
    {
        private readonly GL Gl;
        private uint vao;
        private uint vboPos;
        private uint vboColor;
        private uint shaderProgram;
        private int maxParticles;

        public ParticleRenderer(GL gl, int maxParticles, int screenWidth, int screenHeight)
        {
            this.Gl = gl;
            this.maxParticles = maxParticles;

            InitGraphics(screenWidth, screenHeight);
        }

        private void InitGraphics(int width, int height)
        {
            // VAO erstellen
            vao = Gl.GenVertexArray();
            Gl.BindVertexArray(vao);

            // 1. Position Buffer (VBO)
            vboPos = Gl.GenBuffer();
            Gl.BindBuffer(BufferTargetARB.ArrayBuffer, vboPos);
            Gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(maxParticles * sizeof(Float2)), null, BufferUsageARB.DynamicDraw);
            
            Gl.EnableVertexAttribArray(0); // Location 0
            Gl.VertexAttribPointer(0, 2, VertexAttribPointerType.Float, false, (uint)sizeof(Float2), (void*)0);

            // 2. Color Buffer (VBO)
            vboColor = Gl.GenBuffer();
            Gl.BindBuffer(BufferTargetARB.ArrayBuffer, vboColor);
            Gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(maxParticles * sizeof(Float4)), null, BufferUsageARB.DynamicDraw);

            Gl.EnableVertexAttribArray(1); // Location 1
            Gl.VertexAttribPointer(1, 4, VertexAttribPointerType.Float, false, (uint)sizeof(Float4), (void*)0);

            // Shader kompilieren
            CompileShaders(width, height);
            
            Gl.BindVertexArray(0);
        }

        private void CompileShaders(int width, int height)
        {
            string vertexSource = @"
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec4 aColor;

out vec4 ParticleColor;
uniform mat4 uProjection;

void main()
{
    gl_Position = uProjection * vec4(aPos, 0.0, 1.0);
    ParticleColor = aColor;
}
";
            string fragmentSource = @"
#version 330 core
in vec4 ParticleColor;
out vec4 FragColor;

void main()
{
    FragColor = ParticleColor;
}
";

            uint vs = CreateShader(ShaderType.VertexShader, vertexSource);
            uint fs = CreateShader(ShaderType.FragmentShader, fragmentSource);

            shaderProgram = Gl.CreateProgram();
            Gl.AttachShader(shaderProgram, vs);
            Gl.AttachShader(shaderProgram, fs);
            Gl.LinkProgram(shaderProgram);

            // Projection Matrix setzen (einmalig, da sich Fenstergröße hier im Beispiel nicht ändert - sonst in Render Methode verschieben)
            Gl.UseProgram(shaderProgram);
            var projection = Matrix4x4.CreateOrthographicOffCenter(0, width, height, 0, -1, 1);
            int loc = Gl.GetUniformLocation(shaderProgram, "uProjection");
            Gl.UniformMatrix4(loc, 1, false, (float*)&projection);

            Gl.DeleteShader(vs);
            Gl.DeleteShader(fs);
        }

        private uint CreateShader(ShaderType type, string src)
        {
            uint shader = Gl.CreateShader(type);
            Gl.ShaderSource(shader, src);
            Gl.CompileShader(shader);
            
            string info = Gl.GetShaderInfoLog(shader);
            if (!string.IsNullOrWhiteSpace(info))
            {
                Console.WriteLine($"Shader Error ({type}): {info}");
            }
            return shader;
        }

        public void Render(Float2[] positions, Float4[] colors, int count)
        {
            if (count == 0) return;

            Gl.UseProgram(shaderProgram);
            Gl.BindVertexArray(vao);

            // Update Positionen
            Gl.BindBuffer(BufferTargetARB.ArrayBuffer, vboPos);
            fixed (void* data = positions)
            {
                Gl.BufferSubData(BufferTargetARB.ArrayBuffer, 0, (nuint)(count * sizeof(Float2)), data);
            }

            // Update Farben
            Gl.BindBuffer(BufferTargetARB.ArrayBuffer, vboColor);
            fixed (void* data = colors)
            {
                Gl.BufferSubData(BufferTargetARB.ArrayBuffer, 0, (nuint)(count * sizeof(Float4)), data);
            }

            // Zeichnen
            Gl.DrawArrays(PrimitiveType.Points, 0, (uint)count);
            Gl.BindVertexArray(0);
        }

        public void UpdateProjection(int width, int height)
        {
             Gl.UseProgram(shaderProgram);
             var projection = Matrix4x4.CreateOrthographicOffCenter(0, width, height, 0, -1, 1);
             int loc = Gl.GetUniformLocation(shaderProgram, "uProjection");
             Gl.UniformMatrix4(loc, 1, false, (float*)&projection);
        }

        public void Dispose()
        {
            Gl.DeleteVertexArray(vao);
            Gl.DeleteBuffer(vboPos);
            Gl.DeleteBuffer(vboColor);
            Gl.DeleteProgram(shaderProgram);
        }
    }
}

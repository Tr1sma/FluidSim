using System;
using System.Numerics;
using ComputeSharp;
using NBL.Particles.Shaders;

namespace NBL.Particles
{
    public class FluidParticleSystem : IDisposable
    {
        // Settings
        public int MaxParticles { get; private set; }
        private readonly GraphicsDevice device;

        // GPU Buffers
        private ReadWriteBuffer<Float2> posBuffer;
        private ReadWriteBuffer<Float2> velBuffer;
        private ReadWriteBuffer<Float2> accBuffer;
        private ReadWriteBuffer<Float4> colorBuffer;
        private ReadWriteBuffer<float> lifeTimeBuffer;
        private ReadWriteBuffer<int> typeBuffer;

        // CPU Staging for Rendering (Sync)
        public Float2[] CpuPos { get; private set; }
        public Float4[] CpuColor { get; private set; }

        public int ActiveParticleCount { get; private set; } = 0;
        
        private int screenWidth;
        private int screenHeight;

        public FluidParticleSystem(int maxParticles, int width, int height)
        {
            MaxParticles = maxParticles;
            screenWidth = width;
            screenHeight = height;
            device = GraphicsDevice.GetDefault();

            AllocateBuffers();
        }

        private void AllocateBuffers()
        {
            posBuffer = device.AllocateReadWriteBuffer<Float2>(MaxParticles);
            velBuffer = device.AllocateReadWriteBuffer<Float2>(MaxParticles);
            accBuffer = device.AllocateReadWriteBuffer<Float2>(MaxParticles);
            colorBuffer = device.AllocateReadWriteBuffer<Float4>(MaxParticles);
            lifeTimeBuffer = device.AllocateReadWriteBuffer<float>(MaxParticles);
            typeBuffer = device.AllocateReadWriteBuffer<int>(MaxParticles);

            CpuPos = new Float2[MaxParticles];
            CpuColor = new Float4[MaxParticles];
        }

        public void ResizeScreen(int w, int h)
        {
            screenWidth = w;
            screenHeight = h;
        }

        public void SpawnParticle(Vector2 position, Vector2 velocity, Vector4 color, float lifeTime, int type)
        {
            if (ActiveParticleCount >= MaxParticles) return; // Oder Ring-Buffer Logik implementieren

            int index = ActiveParticleCount;
            
            // Upload single particle data (Langsam für Massen, besser Batched machen!)
            // Für die Engine: Besser eine Methode "SpawnBatch" bauen.
            posBuffer[index] = new Float2(position.X, position.Y);
            velBuffer[index] = new Float2(velocity.X, velocity.Y);
            accBuffer[index] = Float2.Zero;
            colorBuffer[index] = new Float4(color.X, color.Y, color.Z, color.W);
            lifeTimeBuffer[index] = lifeTime;
            typeBuffer[index] = type;

            ActiveParticleCount++;
        }

        public void Update(float deltaTime)
        {
            if (ActiveParticleCount == 0) return;

            // 1. Grid Build & Clear (Nur für Fluid Partikel nötig)
            // ... (Hier käme deine Grid-Logik hin)

            // 2. Forces Calculation
            // ... (Hier käme deine Force-Logik hin)

            // 3. Integration & Move
            device.For(ActiveParticleCount, new UpdateParticlesShader(
                posBuffer, velBuffer, accBuffer, lifeTimeBuffer, typeBuffer,
                deltaTime, screenWidth, screenHeight, ActiveParticleCount
            ));

            // 4. Sync to CPU for Rendering
            // Optimierung: Nur Positionen und Farben kopieren
            posBuffer.CopyTo(CpuPos, 0, 0, ActiveParticleCount);
            colorBuffer.CopyTo(CpuColor, 0, 0, ActiveParticleCount);
        }

        public void Dispose()
        {
            posBuffer.Dispose();
            velBuffer.Dispose();
            accBuffer.Dispose();
            colorBuffer.Dispose();
            lifeTimeBuffer.Dispose();
            typeBuffer.Dispose();
        }
    }
}

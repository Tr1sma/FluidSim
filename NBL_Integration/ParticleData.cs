using System.Numerics;
using ComputeSharp;

namespace NBL.Particles
{
    // Die Daten, die auf der GPU leben
    public struct ParticleData
    {
        public Float2 Position;
        public Float2 Velocity;
        public Float4 Color;    // RGBA
        public float LifeTime;  // In Sekunden (verbleibend)
        public int Type;        // 0 = Simple (Deko), 1 = Fluid/Physik
    }

    public static class ParticleTypes
    {
        public const int Simple = 0; // Keine Kollision, stirbt nach Zeit
        public const int Fluid = 1;  // Kollidiert, permanente Flüssigkeit
    }
}

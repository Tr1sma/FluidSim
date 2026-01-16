using ComputeSharp;

namespace NBL.Particles.Shaders
{
    [ComputeSharp.GeneratedComputeShaderDescriptor]
    [ComputeSharp.ThreadGroupSize(64, 1, 1)]
    public readonly partial struct UpdateParticlesShader : IComputeShader
    {
        public readonly ReadWriteBuffer<Float2> Pos;
        public readonly ReadWriteBuffer<Float2> Vel;
        public readonly ReadWriteBuffer<Float2> Acc;
        public readonly ReadWriteBuffer<float> LifeTime;
        public readonly ReadWriteBuffer<int> Type;

        public readonly float DeltaTime;
        public readonly int Width;
        public readonly int Height;
        public readonly int ParticleCount;

        public UpdateParticlesShader(
            ReadWriteBuffer<Float2> pos,
            ReadWriteBuffer<Float2> vel,
            ReadWriteBuffer<Float2> acc,
            ReadWriteBuffer<float> lifeTime,
            ReadWriteBuffer<int> type,
            float deltaTime,
            int width,
            int height,
            int particleCount)
        {
            Pos = pos;
            Vel = vel;
            Acc = acc;
            LifeTime = lifeTime;
            Type = type;
            DeltaTime = deltaTime;
            Width = width;
            Height = height;
            ParticleCount = particleCount;
        }

        public void Execute()
        {
            int i = ThreadIds.X;
            if (i >= ParticleCount) return;

            // Lebenszeit verringern
            float life = LifeTime[i];
            if (life > 0)
            {
                life -= DeltaTime;
                LifeTime[i] = life;
            }

            // Physik Update
            Float2 v = Vel[i];
            Float2 a = Acc[i];
            Float2 p = Pos[i];

            // Einfache Euler-Integration
            v.X += a.X * DeltaTime;
            v.Y += a.Y * DeltaTime;
            p.X += v.X * DeltaTime;
            p.Y += v.Y * DeltaTime;

            // Wandkollisionen (für alle Partikel)
            const float bounce = -0.5f;
            if (p.X < 0) { p.X = 0; v.X *= bounce; }
            if (p.Y < 0) { p.Y = 0; v.Y *= bounce; }
            if (p.X > Width) { p.X = Width; v.X *= bounce; }
            if (p.Y > Height) { p.Y = Height; v.Y *= bounce; }

            Vel[i] = v;
            Pos[i] = p;
            
            // Reset Beschleunigung
            Acc[i] = Float2.Zero; 
        }
    }

    [ComputeSharp.GeneratedComputeShaderDescriptor]
    [ComputeSharp.ThreadGroupSize(64, 1, 1)]
    public readonly partial struct ClearGridShader : IComputeShader
    {
        public readonly ReadWriteBuffer<int> GridHeads;

        public ClearGridShader(ReadWriteBuffer<int> gridHeads)
        {
            GridHeads = gridHeads;
        }

        public void Execute()
        {
            GridHeads[ThreadIds.X] = -1;
        }
    }

    [ComputeSharp.GeneratedComputeShaderDescriptor]
    [ComputeSharp.ThreadGroupSize(64, 1, 1)]
    public readonly partial struct BuildGridShader : IComputeShader
    {
        public readonly ReadWriteBuffer<int> GridHeads;
        public readonly ReadWriteBuffer<int> NextParticle;
        public readonly ReadWriteBuffer<Float2> Pos;
        public readonly ReadWriteBuffer<int> Type;
        
        public readonly int GridCols;
        public readonly int GridRows;
        public readonly int ParticleCount;
        public readonly int GridCellSize;

        public BuildGridShader(
            ReadWriteBuffer<int> gridHeads,
            ReadWriteBuffer<int> nextParticle,
            ReadWriteBuffer<Float2> pos,
            ReadWriteBuffer<int> type,
            int gridCols,
            int gridRows,
            int particleCount,
            int gridCellSize)
        {
            GridHeads = gridHeads;
            NextParticle = nextParticle;
            Pos = pos;
            Type = type;
            GridCols = gridCols;
            GridRows = gridRows;
            ParticleCount = particleCount;
            GridCellSize = gridCellSize;
        }

        public void Execute()
        {
            int i = ThreadIds.X;
            if (i >= ParticleCount) return;

            // Nur Fluid-Partikel kommen ins Grid für Kollisionen
            if (Type[i] == 0) // 0 = Simple
                return;

            Float2 p = Pos[i];
            int cx = (int)(p.X / GridCellSize);
            int cy = (int)(p.Y / GridCellSize);

            if (cx < 0) cx = 0; else if (cx >= GridCols) cx = GridCols - 1;
            if (cy < 0) cy = 0; else if (cy >= GridRows) cy = GridRows - 1;

            int cellIndex = cy * GridCols + cx;

            int originalHead;
            Hlsl.InterlockedExchange(ref GridHeads[cellIndex], i, out originalHead);
            NextParticle[i] = originalHead;
        }
    }

    [ComputeSharp.GeneratedComputeShaderDescriptor]
    [ComputeSharp.ThreadGroupSize(64, 1, 1)]
    public readonly partial struct CalculateForcesShader : IComputeShader
    {
        public readonly ReadWriteBuffer<Float2> Acc;
        public readonly ReadWriteBuffer<Float2> Pos;
        public readonly ReadWriteBuffer<Float2> Vel;
        public readonly ReadWriteBuffer<int> GridHeads;
        public readonly ReadWriteBuffer<int> NextParticle;
        public readonly ReadWriteBuffer<int> Type;

        public readonly int GridCols;
        public readonly int GridRows;
        public readonly int ParticleCount;
        public readonly int GridCellSize;
        
        // Physik Konstanten
        public readonly float GravityY;
        public readonly float RepulsionForce;
        public readonly float DampingFactor;
        public readonly float CollisionRadius;
        public readonly float ParticleMass;

        public CalculateForcesShader(
            ReadWriteBuffer<Float2> acc,
            ReadWriteBuffer<Float2> pos,
            ReadWriteBuffer<Float2> vel,
            ReadWriteBuffer<int> gridHeads,
            ReadWriteBuffer<int> nextParticle,
            ReadWriteBuffer<int> type,
            int gridCols,
            int gridRows,
            int particleCount,
            int gridCellSize,
            float gravityY,
            float repulsionForce,
            float dampingFactor,
            float collisionRadius,
            float particleMass)
        {
            Acc = acc;
            Pos = pos;
            Vel = vel;
            GridHeads = gridHeads;
            NextParticle = nextParticle;
            Type = type;
            GridCols = gridCols;
            GridRows = gridRows;
            ParticleCount = particleCount;
            GridCellSize = gridCellSize;
            GravityY = gravityY;
            RepulsionForce = repulsionForce;
            DampingFactor = dampingFactor;
            CollisionRadius = collisionRadius;
            ParticleMass = particleMass;
        }

        public void Execute()
        {
            int i = ThreadIds.X;
            if (i >= ParticleCount) return;

            // Basis-Kräfte (Gravitation) gelten für alle
            float forceX = 0;
            float forceY = GravityY * ParticleMass; // F = m * g

            // Wenn es kein Fluid ist, hören wir hier auf (spart extrem Leistung!)
            if (Type[i] == 0) // 0 = Simple
            {
                 Acc[i] = new Float2(0, GravityY);
                 return;
            }

            Float2 myPos = Pos[i];
            Float2 myVel = Vel[i];
            
            // Grid-Nachbarsuche (nur für Fluids)
            int cx = (int)(myPos.X / GridCellSize);
            int cy = (int)(myPos.Y / GridCellSize);

            // Clamp Indices
            if (cx < 0) cx = 0; else if (cx >= GridCols) cx = GridCols - 1;
            if (cy < 0) cy = 0; else if (cy >= GridRows) cy = GridRows - 1;

            int startX = cx > 0 ? cx - 1 : 0;
            int endX = cx < GridCols - 1 ? cx + 1 : GridCols - 1;
            int startY = cy > 0 ? cy - 1 : 0;
            int endY = cy < GridRows - 1 ? cy + 1 : GridRows - 1;

            float collisionRadiusSqr = CollisionRadius * CollisionRadius;

            for (int y = startY; y <= endY; y++)
            {
                int rowOffset = y * GridCols;
                for (int x = startX; x <= endX; x++)
                {
                    int cellIndex = rowOffset + x;
                    int neighborIdx = GridHeads[cellIndex];

                    while (neighborIdx != -1)
                    {
                        if (i != neighborIdx)
                        {
                            // Wir müssen nicht prüfen ob Nachbar Fluid ist, 
                            // da nur Fluids im Grid registriert sind (siehe BuildGridShader)!
                            
                            Float2 neighborPos = Pos[neighborIdx];
                            float offX = myPos.X - neighborPos.X;
                            float offY = myPos.Y - neighborPos.Y;

                            float distSqr = offX * offX + offY * offY;
                            if (distSqr < collisionRadiusSqr && distSqr > 0.0001f)
                            {
                                float distance = Hlsl.Sqrt(distSqr);
                                float factor = (CollisionRadius - distance) / distance * RepulsionForce;

                                forceX += offX * factor;
                                forceY += offY * factor;

                                Float2 neighborVel = Vel[neighborIdx];
                                float relVelX = neighborVel.X - myVel.X;
                                float relVelY = neighborVel.Y - myVel.Y;
                                forceX += relVelX * DampingFactor;
                                forceY += relVelY * DampingFactor;
                            }
                        }
                        neighborIdx = NextParticle[neighborIdx];
                    }
                }
            }

            Acc[i] = new Float2(forceX / ParticleMass, forceY / ParticleMass);
        }
    }
}
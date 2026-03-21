import express from 'express';
import cors from 'cors';
import { createClient } from 'redis';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// ── Redis client ──────────────────────────────────────────────────────────────
let redisClient;
let isRedisConnected = false;
let isConnecting = false;

// ✅ FIXED: Match exact key names written by traffic_monitor.py
const REDIS_KEY_INTERSECTION = "traffic:intersection";
const REDIS_KEY_AMBULANCE    = "traffic:ambulance";
const REDIS_KEY_LANE_PREFIX  = "traffic:lane:";

async function connectRedis() {
    if (isConnecting) return;
    isConnecting = true;
    try {
        if (redisClient) {
            try { await redisClient.disconnect(); } catch (_) {}
        }
        redisClient = createClient({
            username: 'default',
            password: 'Pd37kH1plN8NMSjXtz6shgmHgxYCfTVG',
            socket: {
                host: 'redis-13746.crce283.ap-south-1-2.ec2.cloud.redislabs.com',
                port: 13746,
                connectTimeout: 10000,
                reconnectStrategy: (retries) => {
                    if (retries > 10) return new Error('Too many retries');
                    return Math.min(retries * 100, 3000);
                }
            }
        });
        redisClient.on('error',   (err) => { console.error('Redis Error:', err.message); isRedisConnected = false; });
        redisClient.on('connect', ()    => { console.log('✅ Redis connected'); isRedisConnected = true; });
        redisClient.on('ready',   ()    => { isRedisConnected = true; });
        redisClient.on('end',     ()    => { isRedisConnected = false; });
        await redisClient.connect();
    } catch (err) {
        console.error('❌ Redis connect failed:', err.message);
        isRedisConnected = false;
    } finally {
        isConnecting = false;
    }
}

await connectRedis();

// ── Normalise Python data → UI-expected shape ─────────────────────────────────
//
// Python writes:
//   { lanes: { A: { direction, phases: { left/straight/right:
//       { signal_state, vehicles: {two_wheelers,cars,trucks,heavy} } },
//       total_vehicles }, … }, ambulance_alerts: {…}, last_updated }
//
// We pass it through unchanged but guarantee every lane/phase key exists
// so the frontend never crashes on missing data.
//
function normalisePythonData(raw) {
    const laneNames = ['A', 'B', 'C', 'D'];
    const phases    = ['left', 'straight', 'right'];
    const emptyVehicles = () => ({ two_wheelers: 0, cars: 0, trucks: 0, heavy: 0 });

    const lanes = raw.lanes || {};
    for (const ln of laneNames) {
        if (!lanes[ln]) {
            lanes[ln] = { direction: '', phases: {}, total_vehicles: 0 };
        }
        const lanePhases = lanes[ln].phases || {};
        for (const ph of phases) {
            if (!lanePhases[ph]) {
                lanePhases[ph] = { signal_state: 'RED', vehicles: emptyVehicles() };
            }
            if (!lanePhases[ph].vehicles) {
                lanePhases[ph].vehicles = emptyVehicles();
            }
        }
        lanes[ln].phases = lanePhases;
    }
    return {
        lanes,
        ambulance_alerts: raw.ambulance_alerts || {},
        last_updated:     raw.last_updated      || Date.now() / 1000,
    };
}

// ── /api/traffic ─────────────────────────────────────────────────────────────
app.get('/api/traffic', async (req, res) => {
    try {
        if (!isRedisConnected && !isConnecting) await connectRedis();

        if (isRedisConnected && redisClient) {
            const raw = await redisClient.get(REDIS_KEY_INTERSECTION);

            if (raw) {
                const parsed = JSON.parse(raw);
                const data   = normalisePythonData(parsed);
                return res.json({
                    success:   true,
                    data,
                    timestamp: Date.now(),
                    source:    'redis',
                    key:       REDIS_KEY_INTERSECTION,
                });
            }

            // Redis up but key empty — Python hasn't written yet
            const allKeys = await redisClient.keys('*');
            console.warn('⚠️  No data at key', REDIS_KEY_INTERSECTION, '— available keys:', allKeys);
            return res.json({
                success:   true,
                data:      buildEmptyAllLanes(),
                timestamp: Date.now(),
                source:    'empty',
                debug:     { availableKeys: allKeys },
            });
        }

        // Redis unavailable → mock
        console.warn('⚠️  Redis not connected, serving mock data');
        return res.json({
            success:   true,
            data:      generateMockData(),
            timestamp: Date.now(),
            source:    'mock',
        });

    } catch (err) {
        console.error('API Error:', err);
        return res.status(500).json({
            success: false,
            error:   err.message,
            data:    generateMockData(),
        });
    }
});

// ── /api/ambulance — dedicated ambulance alert endpoint ───────────────────────
app.get('/api/ambulance', async (req, res) => {
    try {
        if (!isRedisConnected && !isConnecting) await connectRedis();
        if (isRedisConnected && redisClient) {
            const raw = await redisClient.get(REDIS_KEY_AMBULANCE);
            return res.json({ success: true, data: raw ? JSON.parse(raw) : { active: false } });
        }
        res.json({ success: false, error: 'Redis not connected' });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
});

// ── /api/lane/:name — individual lane data ────────────────────────────────────
app.get('/api/lane/:name', async (req, res) => {
    const lane = req.params.name.toUpperCase();
    if (!['A','B','C','D'].includes(lane)) {
        return res.status(400).json({ error: 'Invalid lane. Use A, B, C or D.' });
    }
    try {
        if (!isRedisConnected && !isConnecting) await connectRedis();
        if (isRedisConnected && redisClient) {
            const raw = await redisClient.get(`${REDIS_KEY_LANE_PREFIX}${lane}`);
            return res.json({ success: true, lane, data: raw ? JSON.parse(raw) : null });
        }
        res.json({ success: false, error: 'Redis not connected' });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
});

// ── /api/debug/keys — inspect all Redis keys ─────────────────────────────────
app.get('/api/debug/keys', async (req, res) => {
    try {
        if (!isRedisConnected) return res.json({ error: 'Redis not connected' });
        const keys   = await redisClient.keys('*');
        const detail = {};
        for (const k of keys) {
            const type = await redisClient.type(k);
            let value  = null;
            if (type === 'string') {
                const raw = await redisClient.get(k);
                try { value = JSON.parse(raw); } catch { value = raw; }
            }
            detail[k] = { type, value };
        }
        res.json({ keys, detail });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// ── /api/health ───────────────────────────────────────────────────────────────
app.get('/api/health', (req, res) => {
    res.json({
        status:    'healthy',
        redis:     isRedisConnected ? 'connected' : 'disconnected',
        timestamp: Date.now(),
    });
});

// ── Serve frontend ────────────────────────────────────────────────────────────
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ── Helpers ───────────────────────────────────────────────────────────────────
function createEmptyLane(direction = '') {
    return {
        direction,
        phases: {
            left:     { signal_state: 'RED', vehicles: { two_wheelers: 0, cars: 0, trucks: 0, heavy: 0 } },
            straight: { signal_state: 'RED', vehicles: { two_wheelers: 0, cars: 0, trucks: 0, heavy: 0 } },
            right:    { signal_state: 'RED', vehicles: { two_wheelers: 0, cars: 0, trucks: 0, heavy: 0 } },
        },
        total_vehicles: 0,
    };
}

function buildEmptyAllLanes() {
    const dirs = { A: 'UP', B: 'DOWN', C: 'LEFT', D: 'RIGHT' };
    return {
        lanes: Object.fromEntries(
            ['A','B','C','D'].map(ln => [ln, createEmptyLane(dirs[ln])])
        ),
        ambulance_alerts: {},
    };
}

function generateMockData() {
    const randomPhase = () => ({
        signal_state: Math.random() > 0.7 ? 'GREEN' : 'RED',
        vehicles: {
            two_wheelers: Math.floor(Math.random() * 20),
            cars:         Math.floor(Math.random() * 15),
            trucks:       Math.floor(Math.random() * 5),
            heavy:        Math.floor(Math.random() * 3),
        },
    });
    const randomLane = (dir) => ({
        direction: dir,
        phases: { left: randomPhase(), straight: randomPhase(), right: randomPhase() },
        total_vehicles: Math.floor(Math.random() * 40),
    });
    return {
        lanes: {
            A: randomLane('UP'),
            B: randomLane('DOWN'),
            C: randomLane('LEFT'),
            D: randomLane('RIGHT'),
        },
        ambulance_alerts: Math.random() > 0.85 ? {
            A: { detected: true, count: 1, timestamp: Date.now() / 1000,
                 lane: 'A', lane_direction: 'UP',
                 message: 'Emergency vehicle in Lane A heading UP' }
        } : {},
    };
}

// ── Start server ──────────────────────────────────────────────────────────────
const server = app.listen(PORT, () => {
    console.log(`🚀 Server:       http://localhost:${PORT}`);
    console.log(`📡 Traffic API:  http://localhost:${PORT}/api/traffic`);
    console.log(`🚑 Ambulance:    http://localhost:${PORT}/api/ambulance`);
    console.log(`🔍 Debug keys:   http://localhost:${PORT}/api/debug/keys`);
    console.log(`🏥 Health:       http://localhost:${PORT}/api/health`);
    console.log(`\n   Listening for Redis key: "${REDIS_KEY_INTERSECTION}"`);
});

server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
        console.error(`❌ Port ${PORT} in use. Run: lsof -ti:${PORT} | xargs kill -9`);
        process.exit(1);
    } else throw err;
});
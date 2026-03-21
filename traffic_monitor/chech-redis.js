import { createClient } from 'redis';

const client = createClient({
    username: 'default',
    password: 'Pd37kH1plN8NMSjXtz6shgmHgxYCfTVG',
    socket: {
        host: 'redis-13746.crce283.ap-south-1-2.ec2.cloud.redislabs.com',
        port: 13746
    }
});

async function checkRedis() {
    try {
        await client.connect();
        console.log('✅ Connected to Redis Cloud\n');
        
        // Check if key exists
        const exists = await client.exists('traffic_intersection');
        console.log(`Key 'traffic_intersection' exists: ${exists ? 'YES' : 'NO'}\n`);
        
        if (exists) {
            // Get the data
            const data = await client.get('traffic_intersection');
            const trafficData = JSON.parse(data);
            
            console.log('📊 CURRENT REDIS DATA:');
            console.log('=' . repeat(50));
            console.log(JSON.stringify(trafficData, null, 2));
            
            // Check if data is changing
            console.log('\n🔄 Testing for changes...');
            console.log('Press Ctrl+C to stop\n');
            
            // Monitor for changes every 2 seconds
            let lastData = data;
            setInterval(async () => {
                try {
                    const newData = await client.get('traffic_intersection');
                    if (newData !== lastData) {
                        console.log('\n🔔 DATA CHANGED at', new Date().toLocaleTimeString());
                        console.log('New data:', JSON.stringify(JSON.parse(newData), null, 2));
                        lastData = newData;
                    } else {
                        console.log('⏱️ No change at', new Date().toLocaleTimeString());
                    }
                } catch (error) {
                    console.error('Monitor error:', error);
                }
            }, 2000);
            
        } else {
            console.log('❌ No traffic_intersection key found in Redis');
        }
        
    } catch (error) {
        console.error('❌ Error:', error);
    }
}

checkRedis();
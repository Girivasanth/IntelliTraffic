import redis

r = redis.Redis(
    host='redis-13746.crce283.ap-south-1-2.ec2.cloud.redislabs.com',
    port=13746,
    username='default',
    password='Pd37kH1plN8NMSjXtz6shgmHgxYCfTVG',
    decode_responses=True
)

r.delete("traffic_intersection")
r.delete("ambulance_alert")

print("✅ Redis cleared completely")
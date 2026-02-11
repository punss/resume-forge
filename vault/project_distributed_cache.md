# Project: Distributed Caching Sytem

## Overview
Designed and implemented a high-performance distributed caching layer to reduce database load and improve API response times for a high-traffic e-commerce platform.

## Key Metrics
- Reduced average API latency by **40%** (from 200ms to 120ms).
- Decreased database read operations by **60%**.
- Handled peak traffic of **10,000 requests per second** with zero downtime.

## Technical Details
- **Languages**: Go, Python
- **Technologies**: Redis, gRPC, Kubernetes, Prometheus
- Implemented a consistent hashing algorithm to distribute keys evenly across cache nodes.
- Designed a write-through and write-back caching strategy configurable per service.
- Integrated Prometheus metrics to monitor cache hit rates, eviction rates, and latency.

## Challenges
- Dealing with cache stampedes during high-traffic events. Solved by implementing probabilistic early expiration (jitter) and request collapsing.
- ensuring data consistency between the cache and the primary database. Adopted a "delete-on-update" strategy combined with short TTLs for critical data.

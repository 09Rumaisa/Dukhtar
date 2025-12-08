-- Migration: Add endpoint analytics tracking table
-- Purpose: Track API endpoint usage for developer/admin monitoring

CREATE TABLE IF NOT EXISTS endpoint_analytics (
    analytics_id SERIAL PRIMARY KEY,
    endpoint_name VARCHAR(100) UNIQUE NOT NULL,
    hit_count INTEGER DEFAULT 0,
    last_hit TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_endpoint_analytics_name ON endpoint_analytics(endpoint_name);
CREATE INDEX IF NOT EXISTS idx_endpoint_analytics_hit_count ON endpoint_analytics(hit_count DESC);

-- Insert initial record for pregnancy guide generation
INSERT INTO endpoint_analytics (endpoint_name, hit_count, last_hit)
VALUES ('pregnancy_guide_generation', 0, NOW())
ON CONFLICT (endpoint_name) DO NOTHING;

COMMENT ON TABLE endpoint_analytics IS 'Tracks API endpoint usage for system monitoring';
COMMENT ON COLUMN endpoint_analytics.endpoint_name IS 'Name/identifier of the endpoint';
COMMENT ON COLUMN endpoint_analytics.hit_count IS 'Total number of times endpoint was called';
COMMENT ON COLUMN endpoint_analytics.last_hit IS 'Timestamp of most recent endpoint call';

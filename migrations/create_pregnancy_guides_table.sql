-- Migration: Create pregnancy_guides table
-- Date: 2024-12-09
-- Description: Add table to store AI-generated pregnancy guides

CREATE TABLE IF NOT EXISTS pregnancy_guides (
    guide_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    tracking_id INTEGER REFERENCES pregnancy_tracking(tracking_id),
    pregnancy_week INTEGER NOT NULL,
    trimester INTEGER NOT NULL,
    current_weight DECIMAL(5,2),
    pre_pregnancy_weight DECIMAL(5,2),
    height_cm DECIMAL(5,2),
    age INTEGER,
    pre_pregnancy_bmi DECIMAL(4,2),
    weight_gain_kg DECIMAL(5,2),
    activity_level VARCHAR(50),
    dietary_restrictions TEXT,
    medical_conditions TEXT,
    language VARCHAR(20) DEFAULT 'english',
    generated_guide TEXT NOT NULL,
    weight_status VARCHAR(20),
    recommended_weight_gain VARCHAR(20),
    guide_sections JSONB,
    search_queries_used TEXT[],
    generation_status VARCHAR(20) DEFAULT 'completed',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_pregnancy_guides_user_id ON pregnancy_guides(user_id);
CREATE INDEX IF NOT EXISTS idx_pregnancy_guides_tracking_id ON pregnancy_guides(tracking_id);

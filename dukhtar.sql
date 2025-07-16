CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    phone VARCHAR(20),
    number_of_children INTEGER,
    age_of_last_child INTEGER,
    current_weight DECIMAL(5,2),
    ongoing_diseases TEXT,
    additional_info TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Pregnancy tracking table
CREATE TABLE pregnancy_tracking (
    tracking_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    pregnancy_start_date DATE NOT NULL,
    due_date DATE NOT NULL,
    current_week INTEGER,
    current_day INTEGER,
    baby_name VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Health metrics table
CREATE TABLE health_metrics (
    metric_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    tracking_id INTEGER REFERENCES pregnancy_tracking(tracking_id),
    recorded_date DATE NOT NULL,
    weight_kg DECIMAL(5,2),
    blood_pressure_systolic INTEGER,
    blood_pressure_diastolic INTEGER,
    heart_rate INTEGER,
    baby_movements_count INTEGER,
    mood_score INTEGER CHECK (mood_score >= 1 AND mood_score <= 10),
    energy_level INTEGER CHECK (energy_level >= 1 AND energy_level <= 10),
    sleep_hours DECIMAL(3,1),
    symptoms TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);



select * from users;




-- Table for storing doctor profiles
CREATE TABLE doctors (
    doctor_id SERIAL PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20),
    specialization VARCHAR(100) NOT NULL, -- e.g., 'Gynecologist', 'Obstetrician', 'Pediatrician'
    qualification VARCHAR(200) NOT NULL, -- e.g., 'MBBS, MD Gynecology'
    experience_years INTEGER NOT NULL,
    license_number VARCHAR(50) UNIQUE NOT NULL,
    hospital_affiliation VARCHAR(200),
    consultation_fee DECIMAL(10,2) NOT NULL,
    profile_image_url VARCHAR(300),
    bio TEXT,
    languages_spoken VARCHAR(200), -- e.g., 'English, Urdu, Punjabi'
    availability_hours VARCHAR(100), -- e.g., '9 AM - 6 PM'
    rating DECIMAL(3,2) DEFAULT 0.00,
    total_reviews INTEGER DEFAULT 0,
    is_verified BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing consultation appointments
CREATE TABLE consultations (
    consultation_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL, -- References users table
    doctor_id INTEGER NOT NULL,
    appointment_date DATE NOT NULL,
    appointment_time TIME NOT NULL,
    consultation_type VARCHAR(20) NOT NULL, -- 'video', 'chat', 'audio'
    status VARCHAR(20) DEFAULT 'scheduled', -- 'scheduled', 'ongoing', 'completed', 'cancelled'
    duration_minutes INTEGER DEFAULT 30,
    consultation_fee DECIMAL(10,2) NOT NULL,
    payment_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'paid', 'refunded'
    meeting_link VARCHAR(300), -- For video consultations
    notes TEXT, -- Doctor's notes after consultation
    prescription TEXT, -- Any prescriptions given
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id) ON DELETE CASCADE
);

-- Table for storing chat messages during consultations
CREATE TABLE consultation_messages (
    message_id SERIAL PRIMARY KEY,
    consultation_id INTEGER NOT NULL,
    sender_type VARCHAR(10) NOT NULL, -- 'user' or 'doctor'
    sender_id INTEGER NOT NULL,
    message_text TEXT NOT NULL,
    message_type VARCHAR(20) DEFAULT 'text', -- 'text', 'image', 'file'
    file_url VARCHAR(300), -- For image/file messages
    is_read BOOLEAN DEFAULT FALSE,
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (consultation_id) REFERENCES consultations(consultation_id) ON DELETE CASCADE
);

-- Table for storing doctor reviews and ratings
CREATE TABLE doctor_reviews (
    review_id SERIAL PRIMARY KEY,
    doctor_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    consultation_id INTEGER NOT NULL,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    review_text TEXT,
    is_anonymous BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id) ON DELETE CASCADE,
    FOREIGN KEY (consultation_id) REFERENCES consultations(consultation_id) ON DELETE CASCADE,
    UNIQUE(consultation_id) -- One review per consultation
);

-- Table for doctor availability slots
CREATE TABLE doctor_availability (
    availability_id SERIAL PRIMARY KEY,
    doctor_id INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL, -- 1=Monday, 2=Tuesday, etc.
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    is_available BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id) ON DELETE CASCADE
);

-- Table for storing user medical history (for better consultations)
CREATE TABLE user_medical_history (
    history_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    pregnancy_week INTEGER,
    current_medications TEXT,
    allergies TEXT,
    previous_pregnancies INTEGER DEFAULT 0,
    medical_conditions TEXT,
    emergency_contact VARCHAR(100),
    blood_type VARCHAR(5),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for better performance
CREATE INDEX idx_consultations_user_id ON consultations(user_id);
CREATE INDEX idx_consultations_doctor_id ON consultations(doctor_id);
CREATE INDEX idx_consultations_date ON consultations(appointment_date);
CREATE INDEX idx_consultations_status ON consultations(status);
CREATE INDEX idx_messages_consultation_id ON consultation_messages(consultation_id);
CREATE INDEX idx_doctors_specialization ON doctors(specialization);
CREATE INDEX idx_doctors_rating ON doctors(rating);
CREATE INDEX idx_doctor_availability_doctor_id ON doctor_availability(doctor_id);

-- Insert sample doctors with various profile image scenarios
INSERT INTO doctors (
    full_name, email, phone, specialization, qualification, 
    experience_years, license_number, hospital_affiliation, 
    consultation_fee, profile_image_url, bio, languages_spoken, 
    availability_hours, rating, total_reviews, is_verified, is_active
) VALUES 
(
    'Dr. Kashaf ul Ain',
    'kashaf.ain@hospital.com',
    '03001111111',
    'Gynecologist',
    'MBBS, FCPS Gynecology',
    15,
    'PMC-001',
    'Sheikh Zayd Hospital',
    4000.00,
    'C:\Users\rumai\OneDrive\Desktop\Dukhtar\static\doctors_profile\kashaf.jpg',
    'Senior consultant with expertise in maternal-fetal medicine',
    'English, Urdu',
    '8 AM - 4 PM',
    4.8,
    156,
    TRUE,
    TRUE
);
update doctors set profile_image_url='d1.png' where full_name='Dr Ahmed Ali';
INSERT INTO doctors (
    full_name, email, phone, specialization, qualification, 
    experience_years, license_number, hospital_affiliation, 
    consultation_fee, profile_image_url, bio, languages_spoken, 
    availability_hours, rating, total_reviews, is_verified, is_active
) VALUES 
(
    'Dr. Rumaisa Siddiqa',
    'rumaisasaddiqa@gmail.com',
    '03002222222',
    'Obstetrician',
    'MBBS, MD Obstetrics',
    8,
    'PMC-002',
    'Shifa International Hospital',
    3000.00,
    'C:\Users\rumai\OneDrive\Desktop\Dukhtar\static\doctors_profile\pi.jpg',
    'Specialized in high-risk pregnancies and prenatal care',
    'English, Urdu, Punjabi',
    '9 AM - 5 PM',
    4.6,
    89,
    TRUE,
    TRUE
);
INSERT INTO doctors (
    full_name, email, phone, specialization, qualification, 
    experience_years, license_number, hospital_affiliation, 
    consultation_fee, profile_image_url, bio, languages_spoken, 
    availability_hours, rating, total_reviews, is_verified, is_active
) VALUES 
(
    'Dr Ahmed Ali','ahmed@hospital.pk','03003333333','Pediatrician','MBBS, DCH',5,'PMC-003','Children Hospital Lahore',2500.00,'C:\Users\rumai\OneDrive\Desktop\Dukhtar\static\doctors_profile\d1.png','Passionate about child healthcare and development',
    'English, Urdu',
    '10 AM - 6 PM',
    4.2,
    34,
    TRUE,
    TRUE
),
(
    'Dr Zia Mouhyiuddin',
    'zzia@hospital.pk',
    '03003333333',
    'Pediatrician',
    'MBBS, DCH',
    5,
    'PMC-056',
    'Children Hospital Lahore',
    2500.00,
    'C:\Users\rumai\OneDrive\Desktop\Dukhtar\static\doctors_profile\d2.png',
    'Passionate about child healthcare and development',
    'English, Urdu',
    '10 AM - 6 PM',
    4.2,
    34,
    TRUE,
    TRUE
);
-- New table for storing generated pregnancy guides
CREATE TABLE pregnancy_guides (
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
    generated_guide TEXT NOT NULL, -- The AI-generated comprehensive guide
    weight_status VARCHAR(20), -- 'underweight', 'normal', 'overweight', 'obese'
    recommended_weight_gain VARCHAR(20), -- e.g., '11.5-16 kg'
    guide_sections JSONB, -- Store structured sections if needed
    search_queries_used TEXT[], -- Array of search queries used
    generation_status VARCHAR(20) DEFAULT 'completed', -- 'completed', 'failed', 'processing'
    error_message TEXT, -- Store any error messages
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for better performance
CREATE INDEX idx_pregnancy_guides_user_id ON pregnancy_guides(user_id);
CREATE INDEX idx_pregnancy_guides_tracking_id ON pregnancy_guides(tracking_id);
CREATE INDEX idx_pregnancy_guides_week ON pregnancy_guides(pregnancy_week);
CREATE INDEX idx_pregnancy_guides_created_at ON pregnancy_guides(created_at);

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_pregnancy_guides_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update updated_at
CREATE TRIGGER pregnancy_guides_updated_at_trigger
    BEFORE UPDATE ON pregnancy_guides
    FOR EACH ROW
    EXECUTE FUNCTION update_pregnancy_guides_updated_at();

-- Optional: Table for storing guide analytics/usage
CREATE TABLE pregnancy_guide_analytics (
    analytics_id SERIAL PRIMARY KEY,
    guide_id INTEGER REFERENCES pregnancy_guides(guide_id),
    user_id INTEGER REFERENCES users(user_id),
    action_type VARCHAR(50), -- 'viewed', 'downloaded', 'shared'
    session_duration INTEGER, -- in seconds
    device_type VARCHAR(50),
    user_agent TEXT,
    ip_address INET,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for analytics
CREATE INDEX idx_guide_analytics_guide_id ON pregnancy_guide_analytics(guide_id);
CREATE INDEX idx_guide_analytics_user_id ON pregnancy_guide_analytics(user_id);
CREATE INDEX idx_guide_analytics_action_type ON pregnancy_guide_analytics(action_type);
#!/usr/bin/env python3
"""
Database Models and Schema for Deforestation Risk Analysis

This module defines SQLAlchemy models for storing satellite imagery, biodiversity,
carbon stock, and climate data in a PostgreSQL database with PostGIS extension.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
from geoalchemy2 import Geometry
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

Base = declarative_base()

class BaseModel(Base):
    """Abstract base model with common fields."""
    __abstract__ = True
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    

class StudyRegion(BaseModel):
    """Study regions for deforestation analysis."""
    __tablename__ = 'study_regions'
    
    name = Column(String(100), nullable=False, unique=True)
    code = Column(String(20), nullable=False, unique=True)  # e.g., 'amazon_para'
    description = Column(Text)
    country_code = Column(String(3), nullable=False)
    
    # Geographic boundaries
    geometry = Column(Geometry('POLYGON', srid=4326), nullable=False)
    lat_min = Column(Float, nullable=False)
    lat_max = Column(Float, nullable=False)
    lon_min = Column(Float, nullable=False)  
    lon_max = Column(Float, nullable=False)
    
    # Region characteristics
    climate_zone = Column(String(50))
    forest_type = Column(String(50))
    area_km2 = Column(Float)
    
    # Metadata
    is_active = Column(Boolean, default=True)
    

class SatelliteImagery(BaseModel):
    """Satellite imagery data from various sensors."""
    __tablename__ = 'satellite_imagery'
    
    # Location and time
    region_code = Column(String(20), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False) 
    geometry = Column(Geometry('POINT', srid=4326), nullable=False)
    date_acquired = Column(DateTime, nullable=False)
    
    # Satellite metadata
    satellite = Column(String(20), nullable=False)  # 'MODIS', 'Landsat-8', 'Sentinel-2'
    product = Column(String(50), nullable=False)
    quality_flag = Column(Integer, default=0)  # 0=good, 1=marginal, 2=poor
    cloud_cover = Column(Float)  # percentage
    
    # Spectral bands (reflectance values 0-1)
    blue = Column(Float)
    green = Column(Float) 
    red = Column(Float)
    nir = Column(Float)  # Near-infrared
    swir1 = Column(Float)  # Short-wave infrared 1
    swir2 = Column(Float)  # Short-wave infrared 2
    
    # Vegetation indices
    ndvi = Column(Float)  # Normalized Difference Vegetation Index
    ndwi = Column(Float)  # Normalized Difference Water Index
    evi = Column(Float)   # Enhanced Vegetation Index
    savi = Column(Float)  # Soil Adjusted Vegetation Index
    
    # Temperature data (Celsius)
    lst_celsius = Column(Float)  # Land Surface Temperature
    
    # Derived metrics
    ndvi_change = Column(Float)  # Temporal change in NDVI
    ndvi_trend = Column(Float)   # Rolling average trend
    deforestation_risk = Column(String(10))  # 'low', 'medium', 'high'
    
    # Grid information
    grid_lat = Column(Float)
    grid_lon = Column(Float)  
    grid_id = Column(String(50))
    

class BiodiversityData(BaseModel):
    """Species occurrence and biodiversity data from GBIF."""
    __tablename__ = 'biodiversity_data'
    
    # Location and time
    region_code = Column(String(20), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    geometry = Column(Geometry('POINT', srid=4326), nullable=False)
    date_recorded = Column(DateTime)
    year = Column(Integer)
    
    # GBIF identifiers
    gbif_id = Column(String(50), unique=True)
    gbif_species_key = Column(String(50))
    
    # Taxonomic information
    taxonomic_group = Column(String(20), nullable=False)  # 'mammals', 'birds', etc.
    species_name = Column(String(200), nullable=False)
    scientific_name = Column(String(200))
    common_name = Column(String(200))
    family = Column(String(100))
    order = Column(String(100))
    class_name = Column('class', String(100))
    kingdom = Column(String(100))
    
    # Conservation status
    threat_status = Column(String(30))  # IUCN Red List category
    endemic_status = Column(String(20))  # 'endemic', 'near_endemic', 'widespread'
    
    # Occurrence details
    individual_count = Column(Integer, default=1)
    basis_of_record = Column(String(50))
    institution_code = Column(String(50))
    dataset_name = Column(String(200))
    
    # Data quality
    coordinate_precision = Column(Float)  # uncertainty in meters
    coordinate_quality = Column(String(10))  # 'high', 'medium', 'low', 'unknown'
    temporal_quality = Column(String(10))
    
    # Grid information
    grid_lat = Column(Float)
    grid_lon = Column(Float)
    grid_id = Column(String(50))
    

class CarbonStockData(BaseModel):
    """Forest carbon stock data from NASA GEDI."""
    __tablename__ = 'carbon_stock_data'
    
    # Location and time
    region_code = Column(String(20), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    geometry = Column(Geometry('POINT', srid=4326), nullable=False)
    date_acquired = Column(DateTime, nullable=False)
    
    # GEDI metadata
    shot_number = Column(String(50), unique=True)
    beam_type = Column(String(10))  # 'power', 'coverage'
    quality_flag = Column(Integer, default=0)
    sensitivity = Column(Float)
    solar_elevation = Column(Float)
    
    # Elevation and structure
    elevation = Column(Float)  # meters above sea level
    canopy_height = Column(Float)  # meters
    rh25 = Column(Float)  # relative height 25%
    rh50 = Column(Float)  # relative height 50% 
    rh75 = Column(Float)  # relative height 75%
    rh95 = Column(Float)  # relative height 95%
    
    # Canopy metrics
    canopy_cover = Column(Float)  # fraction 0-1
    pai = Column(Float)  # Plant Area Index
    fhd_normal = Column(Float)  # Foliage Height Diversity
    
    # Biomass and carbon (Mg/ha)
    agbd = Column(Float)  # Aboveground Biomass Density
    agbd_se = Column(Float)  # Standard error of AGBD
    total_biomass = Column(Float)  # Above + belowground
    carbon_stock = Column(Float)  # Total carbon stock
    carbon_loss_potential = Column(Float)  # Potential loss if deforested
    
    # Forest type and risk assessment
    forest_type = Column(String(30))
    degradation_risk = Column(String(10))  # 'low', 'medium', 'high'
    
    # Grid information  
    grid_lat = Column(Float)
    grid_lon = Column(Float)
    grid_id = Column(String(50))
    

class ClimateData(BaseModel):
    """Climate data from WorldClim and other sources."""
    __tablename__ = 'climate_data'
    
    # Location
    region_code = Column(String(20), nullable=False)
    latitude = Column(Float, nullable=False) 
    longitude = Column(Float, nullable=False)
    geometry = Column(Geometry('POINT', srid=4326), nullable=False)
    climate_zone = Column(String(30))
    
    # Temperature variables (°C * 10, WorldClim format)
    bio01_annual_mean_temp = Column(Float)  # Annual Mean Temperature
    bio02_mean_diurnal_range = Column(Float)  # Mean Diurnal Range
    bio05_max_temp_warmest = Column(Float)  # Max Temperature of Warmest Month
    bio06_min_temp_coldest = Column(Float)  # Min Temperature of Coldest Month
    bio07_temp_annual_range = Column(Float)  # Temperature Annual Range
    bio08_mean_temp_wettest_quarter = Column(Float)
    bio09_mean_temp_driest_quarter = Column(Float)
    
    # Precipitation variables (mm)
    bio12_annual_precipitation = Column(Float)  # Annual Precipitation
    bio13_precip_wettest_month = Column(Float)  # Precipitation of Wettest Month
    bio14_precip_driest_month = Column(Float)  # Precipitation of Driest Month
    bio15_precip_seasonality = Column(Float)  # Precipitation Seasonality
    bio16_precip_wettest_quarter = Column(Float)  # Precipitation of Wettest Quarter
    bio17_precip_driest_quarter = Column(Float)  # Precipitation of Driest Quarter
    bio18_precip_warmest_quarter = Column(Float)  # Precipitation of Warmest Quarter
    bio19_precip_coldest_quarter = Column(Float)  # Precipitation of Coldest Quarter
    
    # Derived climate indices
    aridity_index = Column(Float)
    growing_degree_days = Column(Float)
    potential_evapotranspiration = Column(Float)
    
    # Risk indicators
    drought_risk = Column(String(10))  # 'low', 'medium', 'high'
    temperature_stress = Column(String(10))
    
    # Grid information
    grid_lat = Column(Float)
    grid_lon = Column(Float)
    grid_id = Column(String(50))
    

class SoilData(BaseModel):
    """Soil properties data from SoilGrids."""
    __tablename__ = 'soil_data'
    
    # Location
    region_code = Column(String(20), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    geometry = Column(Geometry('POINT', srid=4326), nullable=False)
    
    # Depth information
    depth_range = Column(String(20), default='0-30')  # cm
    
    # Soil texture (%)
    sand_content = Column(Float)
    silt_content = Column(Float)
    clay_content = Column(Float)
    texture_class = Column(String(30))  # USDA texture class
    
    # Soil chemistry
    ph_h2o = Column(Float)  # Soil pH in H2O
    soil_organic_carbon = Column(Float)  # g/kg
    total_nitrogen = Column(Float)  # g/kg
    cation_exchange_capacity = Column(Float)  # cmol/kg
    
    # Soil physics
    bulk_density = Column(Float)  # g/cm³
    porosity = Column(Float)  # %
    water_holding_capacity = Column(Float)  # vol%
    infiltration_rate = Column(Float)  # cm/hr
    
    # Derived properties
    erosion_susceptibility = Column(String(10))  # 'low', 'medium', 'high'
    fertility_index = Column(Float)  # 0-1 scale
    degradation_risk = Column(String(10))  # 'low', 'medium', 'high'
    
    # Grid information
    grid_lat = Column(Float)
    grid_lon = Column(Float)
    grid_id = Column(String(50))
    

class IntegratedData(BaseModel):
    """Integrated dataset combining all environmental variables."""
    __tablename__ = 'integrated_data'
    
    # Location and time
    region_code = Column(String(20), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    geometry = Column(Geometry('POINT', srid=4326), nullable=False)
    date_integrated = Column(DateTime, default=datetime.utcnow)
    
    # Grid information
    grid_lat = Column(Float, nullable=False)
    grid_lon = Column(Float, nullable=False)
    grid_id = Column(String(50), nullable=False)
    
    # Satellite metrics (aggregated)
    mean_ndvi = Column(Float)
    ndvi_trend = Column(Float)
    mean_lst = Column(Float)
    satellite_risk_score = Column(Float)
    
    # Biodiversity metrics  
    species_richness = Column(Integer, default=0)
    threatened_species_count = Column(Integer, default=0)
    endemic_species_count = Column(Integer, default=0)
    biodiversity_risk_score = Column(Float)
    
    # Carbon metrics
    mean_carbon_stock = Column(Float)
    mean_canopy_height = Column(Float)
    mean_canopy_cover = Column(Float)
    carbon_risk_score = Column(Float)
    
    # Climate metrics
    mean_temperature = Column(Float)  # °C
    annual_precipitation = Column(Float)  # mm
    climate_stability_score = Column(Float)
    water_stress_index = Column(Float)
    
    # Soil metrics
    soil_ph = Column(Float)
    soil_organic_carbon = Column(Float)
    soil_fertility_index = Column(Float)
    soil_vulnerability_score = Column(Float)
    
    # Integrated risk assessment
    environmental_risk_score = Column(Float, nullable=False)  # 0-1 scale
    deforestation_risk_category = Column(String(10), nullable=False)  # 'low', 'medium', 'high'
    
    # Model predictions (to be populated by ML models)
    predicted_risk_1year = Column(Float)
    predicted_risk_5year = Column(Float)
    predicted_risk_10year = Column(Float)
    model_confidence = Column(Float)
    
    # Sustainable deforestation metrics
    forest_cover_percentage = Column(Float, comment="Current forest cover percentage")
    sustainable_harvest_rate = Column(Float, comment="Maximum sustainable harvest rate (% per year)")
    safe_deforestation_percentage = Column(Float, comment="Safe annual deforestation percentage")
    carbon_buffer_threshold = Column(Float, comment="Minimum carbon stock to maintain (Mg/ha)")
    biodiversity_conservation_requirement = Column(Float, comment="Minimum area for biodiversity conservation (%)")
    ecological_resilience_score = Column(Float, comment="Ecosystem's ability to recover (0-1 scale)")
    
    # Additional metadata
    data_completeness_score = Column(Float)  # Percentage of data fields populated
    quality_score = Column(Float)  # Overall data quality assessment
    

class DataSource(BaseModel):
    """Data source metadata and processing history."""
    __tablename__ = 'data_sources'
    
    name = Column(String(100), nullable=False)
    description = Column(Text)
    url = Column(String(500))
    api_version = Column(String(20))
    
    # Processing information
    last_update = Column(DateTime)
    processing_version = Column(String(20))
    processing_parameters = Column(JSON)  # Store processing config as JSON
    
    # Data quality metrics
    total_records = Column(Integer, default=0)
    valid_records = Column(Integer, default=0)
    quality_score = Column(Float)
    
    # Status
    is_active = Column(Boolean, default=True)
    

class ProcessingLog(BaseModel):
    """Log of data processing operations."""
    __tablename__ = 'processing_logs'
    
    operation_type = Column(String(50), nullable=False)  # 'data_fetch', 'integration', 'ml_training'
    status = Column(String(20), nullable=False)  # 'started', 'completed', 'failed'
    message = Column(Text)
    
    # Processing details
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    duration_seconds = Column(Integer)
    
    # Data processed
    records_processed = Column(Integer, default=0)
    records_created = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    
    # Error information
    error_message = Column(Text)
    error_details = Column(JSON)
    
    # Related entities
    region_codes = Column(JSON)  # List of regions processed
    data_sources = Column(JSON)  # List of data sources used
    

class DeforestationAlert(BaseModel):
    """Deforestation risk alerts generated by the system."""
    __tablename__ = 'deforestation_alerts'
    
    # Location
    region_code = Column(String(20), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    geometry = Column(Geometry('POINT', srid=4326), nullable=False)
    
    # Alert details
    alert_type = Column(String(30), nullable=False)  # 'high_risk', 'biodiversity_threat', 'carbon_loss'
    risk_level = Column(String(10), nullable=False)  # 'medium', 'high', 'critical'
    risk_score = Column(Float, nullable=False)
    
    # Alert triggers
    trigger_factors = Column(JSON)  # List of factors that triggered the alert
    ndvi_decline = Column(Float)
    carbon_loss_estimate = Column(Float)
    species_threat_count = Column(Integer)
    
    # Alert status
    status = Column(String(20), default='active')  # 'active', 'investigating', 'resolved', 'false_positive'
    priority = Column(String(10), default='medium')  # 'low', 'medium', 'high'
    
    # Response tracking
    assigned_to = Column(String(100))
    investigation_notes = Column(Text)
    resolution_date = Column(DateTime)
    

# Database connection and session management
class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, database_url: str):
        """Initialize database connection."""
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        
    def drop_tables(self):
        """Drop all database tables (use with caution)."""
        Base.metadata.drop_all(bind=self.engine)
        
    def get_session(self):
        """Get a database session."""
        return self.SessionLocal()
        
    def execute_sql(self, sql: str):
        """Execute raw SQL statement."""
        with self.engine.connect() as connection:
            result = connection.execute(sql)
            return result.fetchall()


# Utility functions for data operations
def create_spatial_index(engine, table_name: str, geometry_column: str = 'geometry'):
    """Create spatial index on geometry column."""
    index_name = f"idx_{table_name}_{geometry_column}"
    sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} USING GIST ({geometry_column})"
    
    with engine.connect() as connection:
        connection.execute(sql)


def create_database_indexes(db_manager: DatabaseManager):
    """Create all recommended database indexes."""
    
    # Spatial indexes
    spatial_tables = [
        'study_regions', 'satellite_imagery', 'biodiversity_data', 
        'carbon_stock_data', 'climate_data', 'soil_data', 
        'integrated_data', 'deforestation_alerts'
    ]
    
    for table in spatial_tables:
        create_spatial_index(db_manager.engine, table)
    
    # Regular indexes for common queries
    with db_manager.engine.connect() as connection:
        # Region code indexes
        connection.execute("CREATE INDEX IF NOT EXISTS idx_satellite_imagery_region_date ON satellite_imagery(region_code, date_acquired)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_biodiversity_region_species ON biodiversity_data(region_code, species_name)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_carbon_stock_region_date ON carbon_stock_data(region_code, date_acquired)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_integrated_data_region_risk ON integrated_data(region_code, deforestation_risk_category)")
        
        # Grid indexes for spatial queries
        connection.execute("CREATE INDEX IF NOT EXISTS idx_satellite_grid ON satellite_imagery(grid_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_biodiversity_grid ON biodiversity_data(grid_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_carbon_grid ON carbon_stock_data(grid_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_integrated_grid ON integrated_data(grid_id)")
        
        # Alert indexes
        connection.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status_risk ON deforestation_alerts(status, risk_level)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_alerts_created ON deforestation_alerts(created_at)")


# Database initialization script
def initialize_database(database_url: str) -> DatabaseManager:
    """Initialize database with tables and indexes."""
    
    db_manager = DatabaseManager(database_url)
    
    # Create all tables
    db_manager.create_tables()
    
    # Create indexes
    create_database_indexes(db_manager)
    
    # Insert default study regions
    insert_default_study_regions(db_manager)
    
    return db_manager


def insert_default_study_regions(db_manager: DatabaseManager):
    """Insert default study regions into the database."""
    
    regions = [
        {
            'name': 'Amazon Para',
            'code': 'amazon_para',
            'description': 'Amazon rainforest region in Para state, Brazil',
            'country_code': 'BR',
            'lat_min': -4.0, 'lat_max': -3.5,
            'lon_min': -53.0, 'lon_max': -52.5,
            'climate_zone': 'tropical_rainforest',
            'forest_type': 'tropical_rainforest',
            'area_km2': 2775.0
        },
        {
            'name': 'Amazon Acre',
            'code': 'amazon_acre',
            'description': 'Amazon rainforest region in Acre state, Brazil',
            'country_code': 'BR',
            'lat_min': -9.0, 'lat_max': -8.5,
            'lon_min': -71.0, 'lon_max': -70.0,
            'climate_zone': 'tropical_rainforest',
            'forest_type': 'tropical_rainforest',
            'area_km2': 2775.0
        },
        {
            'name': 'Amazon Rondonia',
            'code': 'amazon_rondonia',
            'description': 'Amazon rainforest region in Rondonia state, Brazil',
            'country_code': 'BR',
            'lat_min': -11.8, 'lat_max': -11.2,
            'lon_min': -64.0, 'lon_max': -63.0,
            'climate_zone': 'tropical_rainforest',
            'forest_type': 'tropical_rainforest',
            'area_km2': 2880.0
        },
        {
            'name': 'Cerrado Mato Grosso',
            'code': 'cerrado_mato_grosso',
            'description': 'Cerrado savanna region in Mato Grosso state, Brazil',
            'country_code': 'BR',
            'lat_min': -12.9, 'lat_max': -12.3,
            'lon_min': -55.8, 'lon_max': -55.0,
            'climate_zone': 'tropical_savanna',
            'forest_type': 'cerrado',
            'area_km2': 2688.0
        },
        {
            'name': 'Atlantic Forest SP',
            'code': 'atlantic_forest_sp',
            'description': 'Atlantic Forest region in Sao Paulo state, Brazil',
            'country_code': 'BR',
            'lat_min': -24.5, 'lat_max': -23.9,
            'lon_range': -47.5, 'lon_max': -46.9,
            'climate_zone': 'subtropical',
            'forest_type': 'atlantic_forest',
            'area_km2': 2016.0
        }
    ]
    
    session = db_manager.get_session()
    
    try:
        for region_data in regions:
            # Create polygon geometry from bounding box
            geometry_wkt = f"POLYGON(({region_data['lon_min']} {region_data['lat_min']}, " \
                          f"{region_data['lon_max']} {region_data['lat_min']}, " \
                          f"{region_data['lon_max']} {region_data['lat_max']}, " \
                          f"{region_data['lon_min']} {region_data['lat_max']}, " \
                          f"{region_data['lon_min']} {region_data['lat_min']}))"
            
            # Check if region already exists
            existing = session.query(StudyRegion).filter_by(code=region_data['code']).first()
            
            if not existing:
                region = StudyRegion(
                    name=region_data['name'],
                    code=region_data['code'],
                    description=region_data['description'],
                    country_code=region_data['country_code'],
                    lat_min=region_data['lat_min'],
                    lat_max=region_data['lat_max'],
                    lon_min=region_data['lon_min'],
                    lon_max=region_data['lon_max'],
                    climate_zone=region_data['climate_zone'],
                    forest_type=region_data['forest_type'],
                    area_km2=region_data['area_km2'],
                    geometry=geometry_wkt
                )
                
                session.add(region)
        
        session.commit()
        
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


if __name__ == "__main__":
    # Example usage
    DATABASE_URL = "postgresql://postgres:yourpassword@localhost:5432/deforestation_db"
    
    # Initialize database
    db_manager = initialize_database(DATABASE_URL)
    
    print("Database initialized successfully!")
    print("Tables created:")
    for table in Base.metadata.tables.keys():
        print(f"  - {table}")

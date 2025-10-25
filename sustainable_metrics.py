#!/usr/bin/env python3
"""
Sustainable Deforestation Metrics Calculator

This module calculates safe deforestation percentages and sustainable
harvest rates based on forest characteristics, carbon stocks, biodiversity,
and ecological resilience factors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SustainableDeforestationCalculator:
    """
    Calculates safe deforestation percentages based on multiple environmental factors.
    
    The calculation considers:
    - Forest regeneration capacity
    - Carbon stock thresholds
    - Biodiversity conservation requirements
    - Climate resilience factors
    - Soil quality and erosion risk
    - Existing forest cover percentage
    """
    
    def __init__(self):
        # Forest type regeneration rates (% per year as decimals)
        self.forest_regeneration_rates = {
            'tropical_rainforest': 0.008,    # 0.8% per year - high regeneration
            'temperate_forest': 0.006,       # 0.6% per year - moderate regeneration
            'boreal_forest': 0.003,          # 0.3% per year - slow regeneration
            'dry_forest': 0.005,             # 0.5% per year - moderate-slow regeneration
            'mangrove': 0.007,               # 0.7% per year - good regeneration
            'cerrado': 0.004,                # 0.4% per year - savanna regeneration
            'atlantic_forest': 0.003,        # 0.3% per year - fragmented, slower regeneration
            'unknown': 0.004                 # 0.4% per year - conservative default
        }
        
        # Biodiversity conservation thresholds (minimum % to preserve)
        self.biodiversity_thresholds = {
            'very_high': 0.85,  # >50 species, high endemism
            'high': 0.75,       # 20-50 species, some endemism
            'medium': 0.65,     # 10-20 species
            'low': 0.50,        # <10 species
            'unknown': 0.70     # Precautionary approach
        }
        
        # Carbon stock conservation requirements (minimum % to maintain)
        self.carbon_conservation_thresholds = {
            'very_high': 0.90,  # >300 Mg/ha
            'high': 0.80,       # 200-300 Mg/ha  
            'medium': 0.70,     # 100-200 Mg/ha
            'low': 0.60,        # <100 Mg/ha
            'unknown': 0.75     # Conservative default
        }
    
    def calculate_safe_deforestation_percentage(self, 
                                              forest_cover_pct: float,
                                              carbon_stock: Optional[float] = None,
                                              species_richness: Optional[int] = None,
                                              endemic_species: Optional[int] = None,
                                              threatened_species: Optional[int] = None,
                                              canopy_height: Optional[float] = None,
                                              soil_quality: Optional[float] = None,
                                              climate_resilience: Optional[float] = None,
                                              forest_type: str = 'unknown',
                                              region_fragmentation: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate safe annual deforestation percentage for a given area.
        
        Args:
            forest_cover_pct: Current forest cover percentage (0-100)
            carbon_stock: Carbon stock in Mg/ha
            species_richness: Number of species observed
            endemic_species: Number of endemic species
            threatened_species: Number of threatened species
            canopy_height: Average canopy height in meters
            soil_quality: Soil quality index (0-1)
            climate_resilience: Climate resilience score (0-1)
            forest_type: Type of forest ecosystem
            region_fragmentation: Fragmentation index (0-1, higher = more fragmented)
        
        Returns:
            Dictionary with safe deforestation metrics
        """
        
        # Base regeneration capacity
        base_regeneration = self.forest_regeneration_rates.get(forest_type, 0.004)
        
        # Calculate biodiversity conservation requirement
        biodiversity_category = self._categorize_biodiversity(
            species_richness, endemic_species, threatened_species
        )
        biodiversity_conservation = self.biodiversity_thresholds[biodiversity_category]
        
        # Calculate carbon conservation requirement
        carbon_category = self._categorize_carbon_stock(carbon_stock)
        carbon_conservation = self.carbon_conservation_thresholds[carbon_category]
        
        # Calculate ecological resilience score
        resilience_score = self._calculate_resilience_score(
            canopy_height, soil_quality, climate_resilience, region_fragmentation
        )
        
        # Adjust regeneration rate based on resilience
        adjusted_regeneration = base_regeneration * resilience_score
        
        # Apply conservation constraints
        max_conservation_requirement = max(biodiversity_conservation, carbon_conservation)
        minimum_forest_cover = forest_cover_pct * max_conservation_requirement
        
        # Calculate sustainable harvest rate
        # Can't harvest more than regeneration can replace, and must maintain minimum cover
        sustainable_rate = min(
            adjusted_regeneration,  # Limited by regeneration
            max(0, (forest_cover_pct - minimum_forest_cover) / forest_cover_pct)  # Limited by conservation
        )
        
        # Apply additional safety factors
        safety_factor = 0.7  # 30% safety margin
        safe_deforestation_pct = sustainable_rate * safety_factor
        
        # Ensure safe deforestation doesn't exceed a reasonable maximum (2% annually)
        safe_deforestation_pct = min(safe_deforestation_pct, 0.02)
        
        # Calculate actual area that can be safely harvested
        harvestable_area_pct = (forest_cover_pct * safe_deforestation_pct) if forest_cover_pct > 0 else 0
        
        # Calculate recovery time if over-harvested
        if safe_deforestation_pct > 0:
            recovery_years = 1 / adjusted_regeneration if adjusted_regeneration > 0 else float('inf')
        else:
            recovery_years = float('inf')
        
        return {
            'safe_deforestation_percentage': round(safe_deforestation_pct, 4),
            'sustainable_harvest_rate': round(sustainable_rate, 4),
            'harvestable_area_percentage': round(harvestable_area_pct, 4),
            'minimum_forest_cover_required': round(minimum_forest_cover, 2),
            'biodiversity_conservation_requirement': round(biodiversity_conservation * 100, 1),
            'carbon_conservation_requirement': round(carbon_conservation * 100, 1),
            'ecological_resilience_score': round(resilience_score, 3),
            'estimated_recovery_years': round(recovery_years, 1) if recovery_years != float('inf') else None,
            'forest_regeneration_rate': round(adjusted_regeneration, 4),
            'biodiversity_risk_level': biodiversity_category,
            'carbon_risk_level': carbon_category,
            'conservation_priority': self._determine_conservation_priority(
                safe_deforestation_pct, species_richness, carbon_stock
            )
        }
    
    def _categorize_biodiversity(self, species_richness: Optional[int], 
                               endemic_species: Optional[int], 
                               threatened_species: Optional[int]) -> str:
        """Categorize biodiversity conservation priority."""
        if species_richness is None:
            return 'unknown'
        
        # Weight different factors
        biodiversity_score = species_richness
        
        if endemic_species:
            biodiversity_score += endemic_species * 2  # Endemic species are more important
        
        if threatened_species:
            biodiversity_score += threatened_species * 3  # Threatened species are critical
        
        if biodiversity_score >= 60:
            return 'very_high'
        elif biodiversity_score >= 30:
            return 'high'
        elif biodiversity_score >= 15:
            return 'medium'
        else:
            return 'low'
    
    def _categorize_carbon_stock(self, carbon_stock: Optional[float]) -> str:
        """Categorize carbon stock conservation priority."""
        if carbon_stock is None:
            return 'unknown'
        
        if carbon_stock >= 300:
            return 'very_high'
        elif carbon_stock >= 200:
            return 'high'
        elif carbon_stock >= 100:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_resilience_score(self, canopy_height: Optional[float],
                                  soil_quality: Optional[float],
                                  climate_resilience: Optional[float],
                                  fragmentation: Optional[float]) -> float:
        """Calculate overall ecological resilience score."""
        
        resilience_factors = []
        
        # Canopy height factor (taller = more resilient)
        if canopy_height is not None:
            height_factor = min(1.0, canopy_height / 30.0)  # Normalize to 30m max
            resilience_factors.append(height_factor)
        
        # Soil quality factor
        if soil_quality is not None:
            resilience_factors.append(soil_quality)
        
        # Climate resilience factor
        if climate_resilience is not None:
            resilience_factors.append(climate_resilience)
        
        # Fragmentation penalty (less fragmented = more resilient)
        if fragmentation is not None:
            fragmentation_factor = 1.0 - fragmentation
            resilience_factors.append(fragmentation_factor)
        
        # Calculate weighted average, defaulting to 0.6 if no factors
        if resilience_factors:
            return np.mean(resilience_factors)
        else:
            return 0.6  # Conservative default
    
    def _determine_conservation_priority(self, safe_deforestation_pct: float,
                                       species_richness: Optional[int],
                                       carbon_stock: Optional[float]) -> str:
        """Determine overall conservation priority level."""
        
        # Very low or zero safe deforestation = critical priority
        if safe_deforestation_pct <= 0.001:
            return 'critical'
        elif safe_deforestation_pct <= 0.01:
            return 'very_high'
        elif safe_deforestation_pct <= 0.03:
            return 'high'
        elif safe_deforestation_pct <= 0.05:
            return 'medium'
        else:
            return 'low'
    
    def calculate_regional_sustainability_metrics(self, regional_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate sustainability metrics for an entire region."""
        
        if len(regional_data) == 0:
            return {}
        
        # Calculate individual safe deforestation percentages
        safe_deforestation_list = []
        harvestable_areas = []
        conservation_requirements = []
        
        for _, row in regional_data.iterrows():
            metrics = self.calculate_safe_deforestation_percentage(
                forest_cover_pct=row.get('forest_cover_percentage', 70.0),
                carbon_stock=row.get('mean_carbon_stock'),
                species_richness=row.get('species_richness'),
                endemic_species=row.get('endemic_species_count'),
                threatened_species=row.get('threatened_species_count'),
                canopy_height=row.get('mean_canopy_height'),
                soil_quality=row.get('soil_fertility_index'),
                climate_resilience=row.get('climate_stability_score'),
                forest_type=row.get('forest_type', 'unknown'),
                region_fragmentation=row.get('fragmentation_index')
            )
            
            safe_deforestation_list.append(metrics['safe_deforestation_percentage'])
            harvestable_areas.append(metrics['harvestable_area_percentage'])
            conservation_requirements.append(metrics['minimum_forest_cover_required'])
        
        # Regional aggregated metrics
        return {
            'regional_safe_deforestation_avg': round(np.mean(safe_deforestation_list), 4),
            'regional_safe_deforestation_median': round(np.median(safe_deforestation_list), 4),
            'regional_safe_deforestation_min': round(np.min(safe_deforestation_list), 4),
            'regional_safe_deforestation_max': round(np.max(safe_deforestation_list), 4),
            'regional_harvestable_area_total': round(np.sum(harvestable_areas), 2),
            'regional_conservation_requirement_avg': round(np.mean(conservation_requirements), 2),
            'high_conservation_priority_areas': len([x for x in safe_deforestation_list if x <= 0.01]),
            'sustainable_harvest_areas': len([x for x in safe_deforestation_list if x > 0.02]),
            'total_assessed_areas': len(safe_deforestation_list)
        }

def add_sustainability_metrics_to_dataframe(df: pd.DataFrame, 
                                           calculator: SustainableDeforestationCalculator = None) -> pd.DataFrame:
    """Add sustainability metrics to an integrated dataset DataFrame."""
    
    if calculator is None:
        calculator = SustainableDeforestationCalculator()
    
    # Initialize new columns
    sustainability_columns = [
        'forest_cover_percentage', 'sustainable_harvest_rate', 'safe_deforestation_percentage',
        'carbon_buffer_threshold', 'biodiversity_conservation_requirement', 'ecological_resilience_score'
    ]
    
    for col in sustainability_columns:
        if col not in df.columns:
            df[col] = np.nan
    
    logger.info(f"Calculating sustainability metrics for {len(df)} records...")
    
    for idx, row in df.iterrows():
        # Estimate forest cover from NDVI (simplified approximation)
        forest_cover_pct = 85.0  # Default assumption
        if pd.notna(row.get('mean_ndvi')):
            # Higher NDVI generally indicates more forest cover
            ndvi = row['mean_ndvi']
            if ndvi >= 0.7:
                forest_cover_pct = 90.0
            elif ndvi >= 0.5:
                forest_cover_pct = 75.0
            elif ndvi >= 0.3:
                forest_cover_pct = 50.0
            else:
                forest_cover_pct = 25.0
        
        # Calculate sustainability metrics
        metrics = calculator.calculate_safe_deforestation_percentage(
            forest_cover_pct=forest_cover_pct,
            carbon_stock=row.get('mean_carbon_stock'),
            species_richness=row.get('species_richness'),
            endemic_species=row.get('endemic_species_count'),
            threatened_species=row.get('threatened_species_count'),
            canopy_height=row.get('mean_canopy_height'),
            soil_quality=row.get('soil_fertility_index'),
            climate_resilience=row.get('climate_stability_score'),
            forest_type='tropical_rainforest',  # Default for most regions
            region_fragmentation=None
        )
        
        # Update DataFrame with calculated metrics
        df.at[idx, 'forest_cover_percentage'] = forest_cover_pct
        df.at[idx, 'sustainable_harvest_rate'] = metrics['sustainable_harvest_rate']
        df.at[idx, 'safe_deforestation_percentage'] = metrics['safe_deforestation_percentage']
        df.at[idx, 'carbon_buffer_threshold'] = metrics['minimum_forest_cover_required']
        df.at[idx, 'biodiversity_conservation_requirement'] = metrics['biodiversity_conservation_requirement']
        df.at[idx, 'ecological_resilience_score'] = metrics['ecological_resilience_score']
    
    logger.info("Sustainability metrics calculation completed")
    return df

def generate_sustainability_report(df: pd.DataFrame, region_code: str = None) -> Dict:
    """Generate a comprehensive sustainability report for a dataset."""
    
    if region_code:
        region_df = df[df['region_code'] == region_code].copy()
        title = f"Sustainability Report for {region_code}"
    else:
        region_df = df.copy()
        title = "Overall Sustainability Report"
    
    if len(region_df) == 0:
        return {'error': 'No data available for analysis'}
    
    # Calculate summary statistics
    safe_deforestation = region_df['safe_deforestation_percentage'].dropna()
    sustainable_harvest = region_df['sustainable_harvest_rate'].dropna()
    forest_cover = region_df['forest_cover_percentage'].dropna()
    
    report = {
        'title': title,
        'summary': {
            'total_assessed_areas': len(region_df),
            'areas_with_sustainability_data': len(safe_deforestation),
            'average_forest_cover': round(forest_cover.mean(), 1) if len(forest_cover) > 0 else None,
            'average_safe_deforestation_rate': round(safe_deforestation.mean() * 100, 3) if len(safe_deforestation) > 0 else None,
            'average_sustainable_harvest_rate': round(sustainable_harvest.mean() * 100, 3) if len(sustainable_harvest) > 0 else None
        },
        'conservation_priorities': {
            'critical_areas': len(safe_deforestation[safe_deforestation <= 0.001]),
            'very_high_priority': len(safe_deforestation[(safe_deforestation > 0.001) & (safe_deforestation <= 0.01)]),
            'high_priority': len(safe_deforestation[(safe_deforestation > 0.01) & (safe_deforestation <= 0.03)]),
            'medium_priority': len(safe_deforestation[(safe_deforestation > 0.03) & (safe_deforestation <= 0.05)]),
            'low_priority': len(safe_deforestation[safe_deforestation > 0.05])
        },
        'sustainability_ranges': {
            'no_harvest_recommended': len(safe_deforestation[safe_deforestation <= 0.001]),
            'very_limited_harvest': len(safe_deforestation[(safe_deforestation > 0.001) & (safe_deforestation <= 0.02)]),
            'limited_harvest': len(safe_deforestation[(safe_deforestation > 0.02) & (safe_deforestation <= 0.05)]),
            'moderate_harvest_possible': len(safe_deforestation[safe_deforestation > 0.05])
        },
        'recommendations': []
    }
    
    # Generate recommendations
    if report['conservation_priorities']['critical_areas'] > 0:
        report['recommendations'].append(
            f"URGENT: {report['conservation_priorities']['critical_areas']} areas require immediate protection with zero harvest."
        )
    
    if report['conservation_priorities']['very_high_priority'] > 0:
        report['recommendations'].append(
            f"HIGH PRIORITY: {report['conservation_priorities']['very_high_priority']} areas need strict conservation management."
        )
    
    avg_safe_rate = report['summary']['average_safe_deforestation_rate']
    if avg_safe_rate and avg_safe_rate < 1.0:
        report['recommendations'].append(
            f"Regional average safe deforestation rate is only {avg_safe_rate}% annually - implement strict sustainable forestry practices."
        )
    elif avg_safe_rate and avg_safe_rate > 3.0:
        report['recommendations'].append(
            f"Regional average safe deforestation rate is {avg_safe_rate}% annually - sustainable harvest opportunities exist with proper management."
        )
    
    return report

# Example usage and testing
if __name__ == "__main__":
    # Test the calculator
    calculator = SustainableDeforestationCalculator()
    
    # Example: Amazon rainforest area with high biodiversity and carbon
    test_metrics = calculator.calculate_safe_deforestation_percentage(
        forest_cover_pct=85.0,
        carbon_stock=280.0,  # High carbon stock
        species_richness=45,  # High species richness
        endemic_species=8,    # Some endemic species
        threatened_species=3, # Some threatened species
        canopy_height=28.0,   # Tall canopy
        soil_quality=0.7,     # Good soil
        climate_resilience=0.8, # High climate resilience
        forest_type='tropical_rainforest'
    )
    
    print("Test Sustainability Metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value}")
    
    print(f"\nSafe annual deforestation: {test_metrics['safe_deforestation_percentage']:.3f} ({test_metrics['safe_deforestation_percentage'] * 100:.3f}%)")
    print(f"Conservation priority: {test_metrics['conservation_priority']}")
